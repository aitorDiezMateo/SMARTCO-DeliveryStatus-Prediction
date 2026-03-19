"""
Create (but do not train) the best-performing model objects from Optuna studies:

- XGBoost (`src/07_XGBoost.py`)
- CatBoost (`src/08_CatBoost.py`)
- Bagging + RandomForest (`src/09_Bagging.py`)

This module loads the best hyperparameters from SQLite Optuna storages under
`output/optuna/` and instantiates the corresponding estimators / pipelines.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

######### Paths and constants #########

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data"
TRAIN_RAW_PATH = os.path.join(DATA_PATH, "processed", "train_raw.csv")
TEST_RAW_PATH = os.path.join(DATA_PATH, "processed", "test_raw.csv")
OUTPUT_OPTUNA = ROOT / "output" / "optuna"
N_SPLITS = 5

TARGET_COL = "Delivery Status"
DATE_COL = "order date (DateOrders)"
RANDOM_STATE = 42


def _load_fe_module():
    """Load `pipelines/feature_engineering.py` without relying on `src` package."""
    here = Path(__file__).parent
    fe_path = here / "pipelines" / "feature_engineering.py"
    spec = importlib.util.spec_from_file_location("fe_pipeline_module", fe_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load feature engineering module from {fe_path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_best_params(*, db_path: Path, study_name: str) -> dict[str, Any]:
    storage_url = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError as exc:
        # Provide a helpful message (Optuna's default is just "Record does not exist.").
        try:
            available = [s.study_name for s in optuna.get_all_study_summaries(storage=storage_url)]
        except Exception:  # noqa: BLE001
            available = []
        raise KeyError(
            f"Optuna study '{study_name}' not found in DB: {db_path}. "
            f"Available studies: {available}. "
            "If this is the current config, run the corresponding training/tuning script to create it."
        ) from exc
    return dict(study.best_trial.params)


def build_best_xgboost_pipeline():
    """
    Returns an imblearn Pipeline from `pipelines/feature_engineering.build_pipeline`
    with an XGBClassifier configured with Optuna best params.
    """
    fe_mod = _load_fe_module()
    cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL, date_col=DATE_COL)

    # Follow the current configuration in `src/07_XGBoost.py`.
    # If other studies exist, they are treated as older versions.
    db_path = OUTPUT_OPTUNA / "xgboost_study.db"
    study_name = "xgboost_study"
    best = _load_best_params(db_path=db_path, study_name=study_name)

    sampler_choice = str(best.get("sampler", "none"))
    class_weight_choice = str(best.get("class_weight", "none"))
    smoothing = float(best.get("smoothing", 20.0))

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=1,
        n_estimators=int(best["n_estimators"]),
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        min_child_weight=float(best["min_child_weight"]),
        reg_lambda=float(best["reg_lambda"]),
        reg_alpha=float(best["reg_alpha"]),
        gamma=float(best["gamma"]),
    )

    sampler = fe_mod.make_resampler(
        sampler_choice,
        categorical_cols=list(cfg.low_card_cols) + list(cfg.high_card_cols),
        random_state=RANDOM_STATE,
    )

    pipe = fe_mod.build_pipeline(
        model=model,
        oversampler=sampler,
        smoothing=smoothing,
        classes=None,
        config=cfg,
    )

    return {
        "pipeline": pipe,
        "best_params": best,
        "db_path": str(db_path),
        "study_name": study_name,
        "sampler": sampler_choice,
        "class_weight": class_weight_choice,
        "smoothing": smoothing,
    }


def _build_bagging_from_params(params: dict[str, Any]) -> BaggingClassifier:
    class_weight = None if str(params["estimator_class_weight"]) == "none" else "balanced"
    base_tree = DecisionTreeClassifier(
        max_depth=int(params["estimator_max_depth"]),
        min_samples_split=int(params["estimator_min_samples_split"]),
        min_samples_leaf=int(params["estimator_min_samples_leaf"]),
        ccp_alpha=float(params["estimator_ccp_alpha"]),
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )
    return BaggingClassifier(
        estimator=base_tree,
        n_estimators=int(params["n_estimators"]),
        max_samples=float(params["max_samples"]),
        max_features=float(params["max_features"]),
        bootstrap=True,
        bootstrap_features=bool(params["bootstrap_features"]),
        oob_score=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _build_rf_from_params(params: dict[str, Any]) -> RandomForestClassifier:
    class_weight = None if str(params["rf_class_weight"]) == "none" else "balanced"
    bootstrap = bool(params["rf_bootstrap"])
    return RandomForestClassifier(
        n_estimators=int(params["rf_n_estimators"]),
        max_depth=int(params["rf_max_depth"]),
        min_samples_split=int(params["rf_min_samples_split"]),
        min_samples_leaf=int(params["rf_min_samples_leaf"]),
        ccp_alpha=float(params["rf_ccp_alpha"]),
        class_weight=class_weight,
        max_features=float(params["rf_max_features"]),
        bootstrap=bootstrap,
        oob_score=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_best_bagging_pipeline():
    fe_mod = _load_fe_module()
    cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL, date_col=DATE_COL)

    db_path = OUTPUT_OPTUNA / "bagging_study.db"
    # Follow the current configuration in `src/09_Bagging.py`.
    study_name = "bagging__bagging"
    best = _load_best_params(db_path=db_path, study_name=study_name)

    sampler_choice = str(best.get("sampler", "none"))
    smoothing = float(best.get("smoothing", 20.0))

    model = _build_bagging_from_params(best)
    sampler = fe_mod.make_resampler(
        sampler_choice,
        categorical_cols=list(cfg.low_card_cols) + list(cfg.high_card_cols),
        random_state=RANDOM_STATE,
    )

    pipe = fe_mod.build_pipeline(
        model=model,
        oversampler=sampler,
        smoothing=smoothing,
        classes=None,
        config=cfg,
    )
    return {
        "pipeline": pipe,
        "best_params": best,
        "db_path": str(db_path),
        "study_name": study_name,
        "sampler": sampler_choice,
        "smoothing": smoothing,
    }


def build_best_random_forest_pipeline():
    fe_mod = _load_fe_module()
    cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL, date_col=DATE_COL)

    db_path = OUTPUT_OPTUNA / "bagging_study.db"
    # Follow the current configuration in `src/09_Bagging.py`.
    study_name = "bagging__rf"
    best = _load_best_params(db_path=db_path, study_name=study_name)

    sampler_choice = str(best.get("sampler", "none"))
    smoothing = float(best.get("smoothing", 20.0))

    model = _build_rf_from_params(best)
    sampler = fe_mod.make_resampler(
        sampler_choice,
        categorical_cols=list(cfg.low_card_cols) + list(cfg.high_card_cols),
        random_state=RANDOM_STATE,
    )

    pipe = fe_mod.build_pipeline(
        model=model,
        oversampler=sampler,
        smoothing=smoothing,
        classes=None,
        config=cfg,
    )
    return {
        "pipeline": pipe,
        "best_params": best,
        "db_path": str(db_path),
        "study_name": study_name,
        "sampler": sampler_choice,
        "smoothing": smoothing,
    }


@dataclass(frozen=True)
class CatBoostPrepConfig:
    target_col: str = TARGET_COL
    date_col: str = DATE_COL
    categorical_cols: tuple[str, ...] = (
        "Type",
        "Shipping Mode",
        "Customer Segment",
        "Department Name",
        "Market",
        "Order Region",
        "Order City",
        "Customer City",
        "Category Name",
        "Product Name",
        "Customer Country",
        "Order Country",
        "shipping_route",
    )


def _catboost_prepare_features(df: pd.DataFrame, cfg: CatBoostPrepConfig) -> tuple[pd.DataFrame, list[int]]:
    X = df.copy()
    if cfg.date_col not in X.columns:
        raise ValueError(f"Expected date column '{cfg.date_col}' not found.")

    d = pd.to_datetime(X[cfg.date_col], errors="coerce")
    X["order_year"] = d.dt.year
    X["order_month"] = d.dt.month
    X["order_day"] = d.dt.day
    X["is_weekend"] = d.dt.weekday.isin([5, 6]).astype(int)
    X["day_of_week"] = d.dt.dayofweek
    X["week_of_year"] = d.dt.isocalendar().week.values.astype(int)
    X["quarter"] = d.dt.quarter
    X["is_month_start"] = d.dt.is_month_start.astype(int)
    X["is_month_end"] = d.dt.is_month_end.astype(int)

    if "Order Region" in X.columns and "Market" in X.columns:
        origin = X["Order Region"].astype(str).str.replace(" ", "_", regex=False)
        destination = X["Market"].astype(str).str.replace(" ", "_", regex=False)
        X["shipping_route"] = origin + "_to_" + destination
    if "Customer Country" in X.columns and "Order Country" in X.columns:
        X["is_cross_border"] = (X["Customer Country"] != X["Order Country"]).astype(int)

    X = X.drop(columns=[cfg.date_col])

    inferred_cat_cols = [
        c
        for c in X.columns
        if (
            pd.api.types.is_object_dtype(X[c])
            or pd.api.types.is_string_dtype(X[c])
            or pd.api.types.is_categorical_dtype(X[c])
            or pd.api.types.is_bool_dtype(X[c])
        )
    ]
    cat_cols_present = sorted(set(inferred_cat_cols) | {c for c in cfg.categorical_cols if c in X.columns})
    for c in cat_cols_present:
        X[c] = X[c].astype(str).fillna("<NA>")

    cat_feature_indices = [int(X.columns.get_loc(c)) for c in cat_cols_present]
    return X, cat_feature_indices


class CatBoostAutoCatFeatures(BaseEstimator, ClassifierMixin):
    """
    Thin sklearn-compatible wrapper:
    - applies the same deterministic feature prep as `src/08_CatBoost.py`
    - infers `cat_features` indices from the prepared X in fit()
    - delegates to an internal CatBoostClassifier
    """

    def __init__(self, model: Any, *, prep_cfg: CatBoostPrepConfig | None = None) -> None:
        self.model = model
        self.prep_cfg = prep_cfg or CatBoostPrepConfig()
        self.cat_feature_indices_: list[int] | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X, y):  # noqa: ANN001
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        X_prep, cat_idx = _catboost_prepare_features(X_df, self.prep_cfg)
        self.cat_feature_indices_ = cat_idx
        self.feature_names_ = X_prep.columns.tolist()
        self.model.fit(X_prep, y, cat_features=cat_idx)
        return self

    def predict(self, X):  # noqa: ANN001
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        X_prep, _ = _catboost_prepare_features(X_df, self.prep_cfg)
        pred = np.asarray(self.model.predict(X_prep)).reshape(-1)
        return pred.astype(int)

    def predict_proba(self, X):  # noqa: ANN001
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        X_prep, _ = _catboost_prepare_features(X_df, self.prep_cfg)
        return np.asarray(self.model.predict_proba(X_prep))


def build_best_catboost_model():
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "catboost is not installed in the current environment. "
            "Install it (e.g. `pip install catboost`) to build the CatBoost model."
        ) from exc

    db_path = OUTPUT_OPTUNA / "catboost_study.db"
    best = _load_best_params(db_path=db_path, study_name="catboost_v2")

    task_type = os.environ.get("CATBOOST_TASK_TYPE", "GPU").upper()
    devices = os.environ.get("CATBOOST_DEVICES", "0")

    # NOTE: We preserve the best hyperparameters. Any sampling strategy stored in Optuna
    # params is returned as metadata; actual resampling happens at training time.
    sampler_choice = str(best.get("sampler", "none"))
    class_weight_method = str(best.get("class_weight_method", "none"))

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        task_type=task_type,
        devices=devices,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        iterations=int(best["iterations"]),
        depth=int(best["depth"]),
        learning_rate=float(best["learning_rate"]),
        l2_leaf_reg=float(best["l2_leaf_reg"]),
        random_strength=float(best["random_strength"]),
        bagging_temperature=float(best["bagging_temperature"]),
        border_count=int(best["border_count"]),
    )

    wrapped = CatBoostAutoCatFeatures(model=model, prep_cfg=CatBoostPrepConfig())
    return {"model": wrapped, "raw_model": model, "best_params": best, "sampler": sampler_choice, "class_weight_method": class_weight_method}


def build_best_base_models():
    """
    Convenience: build all best model objects (untrained).
    """
    return {
        "xgboost": build_best_xgboost_pipeline(),
        "catboost": build_best_catboost_model(),
        "bagging": build_best_bagging_pipeline(),
        "random_forest": build_best_random_forest_pipeline(),
    }

######### Load training data #########
df = pd.read_csv(TRAIN_RAW_PATH)
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {TRAIN_RAW_PATH}")
if DATE_COL not in df.columns:
    raise ValueError(f"Date column '{DATE_COL}' not found in {TRAIN_RAW_PATH}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
if df[DATE_COL].isna().any():
    raise ValueError(
        f"Date column '{DATE_COL}' contains invalid values after parsing. "
        "Please ensure it can be parsed as datetime."
    )

df = df.sort_values(DATE_COL).reset_index(drop=True)

y_raw = df[TARGET_COL].astype(str)
X = df.drop(columns=[TARGET_COL])

######### Load testing data #########

df_test = pd.read_csv(TEST_RAW_PATH)
if TARGET_COL not in df_test.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {TEST_RAW_PATH}")
if DATE_COL not in df_test.columns:
    raise ValueError(f"Date column '{DATE_COL}' not found in {TEST_RAW_PATH}")

df_test[DATE_COL] = pd.to_datetime(df_test[DATE_COL], errors="coerce")
if df_test[DATE_COL].isna().any():
    raise ValueError(
        f"Date column '{DATE_COL}' contains invalid values after parsing. "
        "Please ensure it can be parsed as datetime."
    )

df_test = df_test.sort_values(DATE_COL).reset_index(drop=True)

X_test = df_test.drop(columns=[TARGET_COL])
y_test_raw = df_test[TARGET_COL].astype(str)

######### Encode target #########

le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = int(np.unique(y).size)

######### Cross-validation configuration #########

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
cv_splits = list(tscv.split(X))

######### Build Stacked Base Models #########

base_models = build_best_base_models()

######### Build StackingClassifier #########
stacking_model = StackingClassifier(
    estimators=[
        ("xgboost", base_models["xgboost"]["pipeline"]),
        ("catboost", base_models["catboost"]["model"]),
        ("bagging", base_models["bagging"]["pipeline"]),
        ("random_forest", base_models["random_forest"]["pipeline"]),
    ],
    final_estimator=LogisticRegression(
        random_state=RANDOM_STATE,
    ),
    cv = tscv,
    stack_method="predict_proba",
)

######### Train stacking model #########
stacking_model.fit(X, y)

######### Evaluate stacking model #########
y_test = le.transform(y_test_raw)
y_pred = stacking_model.predict(X_test)

try:
    y_proba = stacking_model.predict_proba(X_test)
except Exception:  # noqa: BLE001
    y_proba = None

print("=== Classification report (macro/weighted are imbalance-friendly) ===")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[str(c) for c in le.classes_],
        digits=4,
        zero_division=0,
    )
)

print(f"MCC: {matthews_corrcoef(y_test, y_pred):.6f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.6f}")
print(f"F1 macro: {f1_score(y_test, y_pred, average='macro'):.6f}")
print(f"F1 weighted: {f1_score(y_test, y_pred, average='weighted'):.6f}")

if y_proba is not None:
    print("=== Probabilistic metrics (one-vs-rest for multiclass) ===")
    print(f"ROC-AUC macro (OVR): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'):.6f}")
    print(f"ROC-AUC weighted (OVR): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'):.6f}")

    y_true_ovr = np.eye(n_classes, dtype=int)[y_test]
    pr_auc_macro = average_precision_score(y_true_ovr, y_proba, average="macro")
    pr_auc_weighted = average_precision_score(y_true_ovr, y_proba, average="weighted")
    print(f"PR-AUC macro (OVR): {pr_auc_macro:.6f}")
    print(f"PR-AUC weighted (OVR): {pr_auc_weighted:.6f}")
