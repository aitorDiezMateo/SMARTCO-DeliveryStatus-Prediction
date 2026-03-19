"""
This script runs Optuna hyperparameter tuning for GPU XGBoost with CV-safe
feature engineering and time-aware cross-validation.

It loads `data/processed/train_raw.csv`, builds an imblearn Pipeline with
fold-safe encoders, and tunes XGBoost hyperparameters using TimeSeriesSplit.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
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
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
import importlib.util


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


######### Paths and constants #########

DATA_PATH = Path(__file__).parent.parent / "data"
TRAIN_RAW_PATH = os.path.join(DATA_PATH, "processed", "train_raw.csv")
TEST_RAW_PATH = os.path.join(DATA_PATH, "processed", "test_raw.csv")
OUTPUT_PATH = Path(__file__).parent.parent / "output" / "optuna"

TARGET_COL = "Delivery Status"
DATE_COL = "order date (DateOrders)"
STUDY_NAME = "xgboost_study"
N_TRIALS = 200
DB_PATH = OUTPUT_PATH / "xgboost_study.db"
N_SPLITS = 5


######### Load training data #########

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
storage_url = f"sqlite:///{DB_PATH}"

print(f"Optuna storage: {storage_url}")
print(f"Study name:     {STUDY_NAME}")
print(f"N trials:       {N_TRIALS}")
print()

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

######### Encode target #########

le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = int(np.unique(y).size)

######### Load feature engineering pipeline module #########

fe_mod = _load_fe_module()
cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL)

######### Cross-validation configuration #########

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
cv_splits = list(tscv.split(X))


######### Optuna objective #########

def objective(trial: optuna.Trial) -> float:
    sampler_choice = trial.suggest_categorical(
        "sampler",
        [
            "none",
            "random_over",
            "random_under",
            "smotenc",
            "smotenc_tomek",
            "smotenc_enn",
        ],
    )

    if sampler_choice == "none":
        class_weight_choice = trial.suggest_categorical(
            "class_weight", ["none", "balanced"],
        )
    else:
        class_weight_choice = "none"

    smoothing = trial.suggest_float("smoothing", 5.0, 100.0, log=True)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
    }

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        random_state=42,
        n_jobs=1,
        **params,
    )

    sampler = fe_mod.make_resampler(
        sampler_choice,
        categorical_cols=list(cfg.low_card_cols) + list(cfg.high_card_cols),
        random_state=42,
    )

    pipe = fe_mod.build_pipeline(
        model=model,
        oversampler=sampler,
        smoothing=smoothing,
        classes=list(range(n_classes)),
        config=cfg,
    )

    fold_scores: list[float] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        fit_kwargs = {}
        if class_weight_choice == "balanced":
            fit_kwargs["model__sample_weight"] = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )

        pipe.fit(X_train, y_train, **fit_kwargs)
        y_pred = pipe.predict(X_valid)

        score = float(f1_score(y_valid, y_pred, average="macro"))
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

test_df = pd.read_csv(TEST_RAW_PATH)
if TARGET_COL not in test_df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {TEST_RAW_PATH}")

X_train_full = X
y_train_full_raw = y_raw

X_test = test_df.drop(columns=[TARGET_COL])
y_test_raw = test_df[TARGET_COL].astype(str)

le_final = LabelEncoder()
y_train_full = le_final.fit_transform(y_train_full_raw)

try:
    y_test = le_final.transform(y_test_raw)
except ValueError as exc:
    raise ValueError(
        "Test set contains labels not seen in training set. "
        "This usually indicates an issue with the temporal split or label cleaning."
    ) from exc

best = study.best_trial
best_params = best.params

best_sampler = best_params.get("sampler", "none")
best_class_weight = best_params.get("class_weight", "none")
best_smoothing = best_params.get("smoothing", 20.0)

final_sampler = fe_mod.make_resampler(
    best_sampler,
    categorical_cols=list(cfg.low_card_cols) + list(cfg.high_card_cols),
    random_state=42,
)

final_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda",
    random_state=42,
    n_jobs=1,
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    min_child_weight=best_params["min_child_weight"],
    reg_lambda=best_params["reg_lambda"],
    reg_alpha=best_params["reg_alpha"],
    gamma=best_params["gamma"],
)

final_pipe = fe_mod.build_pipeline(
    model=final_model,
    oversampler=final_sampler,
    smoothing=best_smoothing,
    classes=list(range(n_classes)),
    config=cfg,
)

fit_kwargs = {}
if best_class_weight == "balanced" and best_sampler == "none":
    fit_kwargs["model__sample_weight"] = compute_sample_weight(
        class_weight="balanced",
        y=y_train_full,
    )

final_pipe.fit(X_train_full, y_train_full, **fit_kwargs)
test_pred = final_pipe.predict(X_test)

print("\nTest evaluation (best trial trained on full train):")
print(f"  accuracy:        {accuracy_score(y_test, test_pred):.4f}")
print(f"  f1_macro:        {f1_score(y_test, test_pred, average='macro'):.4f}")
print(f"  f1_weighted:     {f1_score(y_test, test_pred, average='weighted'):.4f}")
print(f"  precision_macro: {precision_score(y_test, test_pred, average='macro'):.4f}")
print(f"  recall_macro:    {recall_score(y_test, test_pred, average='macro'):.4f}")

try:
    test_proba = final_pipe.predict_proba(X_test)
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
        target_names=le_final.classes_.tolist(),
        digits=4,
        zero_division=0,
    )
)
