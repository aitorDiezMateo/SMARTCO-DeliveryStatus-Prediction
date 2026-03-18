"""
This script runs Optuna hyperparameter tuning for CatBoost with time-aware
cross-validation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


@dataclass(frozen=True)
class Config:
    target_col: str = "Delivery Status"
    date_col: str = "order date (DateOrders)"

    # CatBoost can handle high-cardinality categoricals directly, but it must know
    # which columns are categorical.
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

RANDOM_STATE = 42


def _add_calendar_features(df: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[date_col], errors="coerce")
    if d.isna().any():
        raise ValueError(f"Invalid datetime values found in '{date_col}'.")

    df["order_year"] = d.dt.year
    df["order_month"] = d.dt.month
    df["order_day"] = d.dt.day
    df["is_weekend"] = d.dt.weekday.isin([5, 6]).astype(int)
    df["day_of_week"] = d.dt.dayofweek
    df["week_of_year"] = d.dt.isocalendar().week.values.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    return df


def _add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Order Region" in df.columns and "Market" in df.columns:
        origin = df["Order Region"].astype(str).str.replace(" ", "_", regex=False)
        destination = df["Market"].astype(str).str.replace(" ", "_", regex=False)
        df["shipping_route"] = origin + "_to_" + destination
    if "Customer Country" in df.columns and "Order Country" in df.columns:
        df["is_cross_border"] = (df["Customer Country"] != df["Order Country"]).astype(int)
    return df


def _prepare_features(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[int]]:
    """
    Returns:
      X: DataFrame with engineered features and cleaned categoricals.
      cat_feature_indices: indices for CatBoost 'cat_features'.
    """
    if cfg.date_col not in df.columns:
        raise ValueError(f"Expected date column '{cfg.date_col}' not found.")

    X = df.copy()
    X = _add_calendar_features(X, date_col=cfg.date_col)
    X = _add_route_features(X)

    # Keep the raw date column out of the model (engineered features replace it).
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


def _encode_categoricals_for_sampling(
    X: pd.DataFrame, *, cat_features: list[int]
) -> tuple[np.ndarray, list[list[str]]]:
    """
    Encode cat columns to integer codes (fit on X only). Returns:
      - encoded numpy array
      - per-cat-col categories list for inverse_transform (same order as cat_features)
    """
    X_enc = X.copy()
    categories: list[list[str]] = []
    for idx in cat_features:
        col = X_enc.columns[idx]
        s = X_enc[col].astype(str).fillna("<NA>")
        cat = pd.Categorical(s)
        X_enc[col] = cat.codes.astype(int)
        categories.append([str(v) for v in cat.categories.tolist()])
    return X_enc.to_numpy(), categories


def _decode_categoricals_after_sampling(
    X_arr: np.ndarray,
    *,
    columns: list[str],
    cat_features: list[int],
    categories: list[list[str]],
) -> pd.DataFrame:
    X_df = pd.DataFrame(X_arr, columns=columns)
    for i, idx in enumerate(cat_features):
        col = columns[idx]
        cats = categories[i]
        codes = pd.to_numeric(X_df[col], errors="coerce").fillna(-1).astype(int).to_numpy()
        # Map unknown/out-of-range to "<NA>" for CatBoost stability.
        out = np.full(len(codes), "<NA>", dtype=object)
        valid = (codes >= 0) & (codes < len(cats))
        if valid.any():
            out[valid] = np.array(cats, dtype=object)[codes[valid]]
        X_df[col] = out.astype(str)
    return X_df


def _make_sampler(choice: str, *, cat_features: list[int]):
    if choice == "none":
        return None
    if choice == "random_over":
        return RandomOverSampler(random_state=RANDOM_STATE)
    if choice == "random_under":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    if choice == "smote":
        return SMOTENC(categorical_features=cat_features, random_state=RANDOM_STATE)
    if choice == "smote_tomek":
        return SMOTETomek(
            smote=SMOTENC(categorical_features=cat_features, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        )
    if choice == "smote_enn":
        return SMOTEENN(
            smote=SMOTENC(categorical_features=cat_features, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown sampler choice: {choice}")


######### Paths and constants #########

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed"
TRAIN_RAW_PATH = DATA_PATH / "train_raw.csv"
TEST_RAW_PATH = DATA_PATH / "test_raw.csv"
OUTPUT_PATH = ROOT / "output" / "optuna"

CFG = Config()
STUDY_NAME = "catboost_v2"
N_TRIALS = 200
N_SPLITS = int(os.environ.get("CV_N_SPLITS", "5"))
DB_PATH = OUTPUT_PATH / "catboost_study.db"

# Default to GPU (CUDA). Override with CATBOOST_TASK_TYPE=CPU if needed.
TASK_TYPE = os.environ.get("CATBOOST_TASK_TYPE", "GPU").upper()
if TASK_TYPE not in {"CPU", "GPU"}:
    raise ValueError("CATBOOST_TASK_TYPE must be 'CPU' or 'GPU'.")
DEVICES = os.environ.get("CATBOOST_DEVICES", "0")


######### Load training data #########

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
storage_url = f"sqlite:///{DB_PATH}"

print(f"Optuna storage: {storage_url}")
print(f"Study name:     {STUDY_NAME}")
print(f"N trials:       {N_TRIALS}")
print(f"CV splits:      {N_SPLITS} (TimeSeriesSplit)")
print(f"CatBoost:       task_type={TASK_TYPE} devices={DEVICES}")
print()

train_df = pd.read_csv(TRAIN_RAW_PATH)
test_df = pd.read_csv(TEST_RAW_PATH)

if CFG.target_col not in train_df.columns:
    raise ValueError(f"Target column '{CFG.target_col}' not found in {TRAIN_RAW_PATH}")
if CFG.target_col not in test_df.columns:
    raise ValueError(f"Target column '{CFG.target_col}' not found in {TEST_RAW_PATH}")
if CFG.date_col not in train_df.columns:
    raise ValueError(f"Date column '{CFG.date_col}' not found in {TRAIN_RAW_PATH}")

train_df[CFG.date_col] = pd.to_datetime(train_df[CFG.date_col], errors="coerce")
if train_df[CFG.date_col].isna().any():
    raise ValueError(
        f"Date column '{CFG.date_col}' contains invalid values after parsing. "
        "Please ensure it can be parsed as datetime."
    )

train_df = train_df.sort_values(CFG.date_col).reset_index(drop=True)

y_train_raw = train_df[CFG.target_col].astype(str)
X_train_raw = train_df.drop(columns=[CFG.target_col])

y_test_raw = test_df[CFG.target_col].astype(str)
X_test_raw = test_df.drop(columns=[CFG.target_col])

######### Encode target #########

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

try:
    y_test = le.transform(y_test_raw)
except ValueError as exc:
    raise ValueError(
        "Test set contains labels not seen in training set. "
        "This usually indicates an issue with the temporal split or label cleaning."
    ) from exc

n_classes = int(np.unique(y_train).size)

X_train, cat_idx = _prepare_features(X_train_raw, CFG)
X_test, _ = _prepare_features(X_test_raw, CFG)

######### Cross-validation configuration #########

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
cv_splits = list(tscv.split(X_train))


######### Optuna objective #########

def objective(trial: optuna.Trial) -> float:
    sampler_choice = trial.suggest_categorical(
        "sampler",
        [
            "none",
            "random_over",
            "random_under",
            "smote",
            "smote_tomek",
            "smote_enn",
        ],
    )
    if sampler_choice == "none":
        class_weight_choice = trial.suggest_categorical(
            "class_weight_method",
            [
                "none",
                "compute_balanced",
                "auto_balanced",
                "auto_sqrt_balanced",
            ],
        )
    else:
        # Avoid mixing resampling + class weights unless explicitly needed.
        class_weight_choice = "none"

    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 50.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 64, 255),
    }

    fold_scores: list[float] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train.iloc[valid_idx]
        y_va = y_train[valid_idx]

        sampler = _make_sampler(sampler_choice, cat_features=cat_idx)
        if sampler is not None:
            # For SMOTENC / SMOTE* we encode categoricals to integer codes first.
            if sampler_choice in {"smote", "smote_tomek", "smote_enn"}:
                X_tr_arr, cats = _encode_categoricals_for_sampling(X_tr, cat_features=cat_idx)
                X_res_arr, y_res = sampler.fit_resample(X_tr_arr, y_tr)
                X_tr = _decode_categoricals_after_sampling(
                    X_res_arr,
                    columns=X_tr.columns.tolist(),
                    cat_features=cat_idx,
                    categories=cats,
                )
                y_tr = np.asarray(y_res)
            else:
                X_res, y_res = sampler.fit_resample(X_tr, y_tr)
                X_tr = pd.DataFrame(X_res, columns=X_train.columns)
                y_tr = np.asarray(y_res)

        class_weights = None
        auto_class_weights = None
        if class_weight_choice == "compute_balanced":
            classes = np.unique(y_tr)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
            class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
        elif class_weight_choice == "auto_balanced":
            auto_class_weights = "Balanced"
        elif class_weight_choice == "auto_sqrt_balanced":
            auto_class_weights = "SqrtBalanced"

        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            task_type=TASK_TYPE,
            devices=DEVICES,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            class_weights=class_weights,
            auto_class_weights=auto_class_weights,
            **params,
        )

        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_idx,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=100,
        )

        y_pred = np.array(model.predict(X_va)).reshape(-1).astype(int)
        score = float(f1_score(y_va, y_pred, average="macro"))
        fold_scores.append(score)

        trial.report(float(np.mean(fold_scores)), step=fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    trial.set_user_attr("cv_fold_scores", fold_scores)
    trial.set_user_attr("cv_mean_f1_macro", mean_score)
    trial.set_user_attr("cv_std_f1_macro", std_score)

    return mean_score


######### Create and run Optuna study #########

tpe_sampler = TPESampler(seed=42)
pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)

study = optuna.create_study(
    direction="maximize",
    study_name=STUDY_NAME,
    storage=storage_url,
    load_if_exists=True,
    sampler=tpe_sampler,
    pruner=pruner,
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
    callbacks=[
        optuna.study.MaxTrialsCallback(N_TRIALS, states=(optuna.trial.TrialState.COMPLETE,)),
    ],
)

print("Best CV f1_macro:", study.best_value)
print("Best params:")
for k, v in sorted(study.best_params.items()):
    print(f"  {k}: {v}")

pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f"\nTrials: {complete} complete, {pruned} pruned")


######### Train best configuration on full train and evaluate on test #########

best = study.best_trial
best_params = best.params

best_sampler = best_params.get("sampler", "none")
best_class_weight_method = best_params.get("class_weight_method", "none")
final_class_weights = None
final_auto_class_weights = None
if best_sampler == "none":
    if best_class_weight_method == "compute_balanced":
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        final_class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    elif best_class_weight_method == "auto_balanced":
        final_auto_class_weights = "Balanced"
    elif best_class_weight_method == "auto_sqrt_balanced":
        final_auto_class_weights = "SqrtBalanced"

X_train_final = X_train
y_train_final = y_train
final_sampler = _make_sampler(best_sampler, cat_features=cat_idx)
if final_sampler is not None:
    if best_sampler in {"smote", "smote_tomek", "smote_enn"}:
        X_arr, cats = _encode_categoricals_for_sampling(X_train_final, cat_features=cat_idx)
        X_res_arr, y_res = final_sampler.fit_resample(X_arr, y_train_final)
        X_train_final = _decode_categoricals_after_sampling(
            X_res_arr,
            columns=X_train_final.columns.tolist(),
            cat_features=cat_idx,
            categories=cats,
        )
        y_train_final = np.asarray(y_res)
    else:
        X_res, y_res = final_sampler.fit_resample(X_train_final, y_train_final)
        X_train_final = pd.DataFrame(X_res, columns=X_train.columns)
        y_train_final = np.asarray(y_res)

final_model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1",
    task_type=TASK_TYPE,
    devices=DEVICES,
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
    class_weights=final_class_weights,
    auto_class_weights=final_auto_class_weights,
    iterations=int(best_params["iterations"]),
    depth=int(best_params["depth"]),
    learning_rate=float(best_params["learning_rate"]),
    l2_leaf_reg=float(best_params["l2_leaf_reg"]),
    random_strength=float(best_params["random_strength"]),
    bagging_temperature=float(best_params["bagging_temperature"]),
    border_count=int(best_params["border_count"]),
)

final_model.fit(X_train_final, y_train_final, cat_features=cat_idx)
test_pred = np.array(final_model.predict(X_test)).reshape(-1).astype(int)

print("\nTest evaluation (best trial trained on full train):")
print(f"  accuracy:        {accuracy_score(y_test, test_pred):.4f}")
print(f"  f1_macro:        {f1_score(y_test, test_pred, average='macro'):.4f}")
print(f"  f1_weighted:     {f1_score(y_test, test_pred, average='weighted'):.4f}")
print(f"  precision_macro: {precision_score(y_test, test_pred, average='macro'):.4f}")
print(f"  recall_macro:    {recall_score(y_test, test_pred, average='macro'):.4f}")

try:
    test_proba = final_model.predict_proba(X_test)
    roc = roc_auc_score(
        y_test,
        test_proba,
        multi_class="ovr",
        average="macro",
    )
    print(f"  roc_auc_macro:   {roc:.4f}")
except Exception as exc:  # noqa: BLE001
    print(f"  roc_auc_macro:   <not available> ({exc})")

print("\nConfusion matrix (labels in LabelEncoder order):")
print(confusion_matrix(y_test, test_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=le.classes_.tolist(),
        digits=4,
        zero_division=0,
    )
)
