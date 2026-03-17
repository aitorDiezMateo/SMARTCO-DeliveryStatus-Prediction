"""
This script runs Optuna hyperparameter tuning for GPU XGBoost with CV-safe
feature engineering.

It loads `data/processed/train_raw.csv`, builds an sklearn Pipeline with
fold-safe encoders, and tunes XGBoost hyperparameters using Stratified CV.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import xgboost as xgb
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import importlib.util


def _load_fe_module():
    """Load `pipelines/feature_engineering.py` without relying on `src` package."""
    here = Path(__file__).parent
    fe_path = here / "pipelines" / "feature_engineering.py"
    spec = importlib.util.spec_from_file_location("fe_pipeline_module", fe_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load feature engineering module from {fe_path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so decorators like @dataclass can resolve __module__
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
STUDY_NAME = "xgboost"
N_TRIALS = 10
DB_PATH = OUTPUT_PATH / "xgboost_study.db"
N_SPLITS = 3


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

# Time-aware CV requires time-ordered rows
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

# Walk-forward time series CV (train is always strictly earlier than validation).
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
cv_splits = list(tscv.split(X))


######### Optuna objective #########

def objective(trial: optuna.Trial) -> float:
        scaler_choice = trial.suggest_categorical(
            "scaler",
            [
                "none",
                "standard",
                "minmax",
            ],
        )

        sampler_choice = trial.suggest_categorical(
            "sampler",
            [
                "none",
                "random_over",
                "random_under",
                "adasyn",
                "smote",
                "smote_tomek",
                "smote_enn",
            ],
        )

        class_weight_choice = trial.suggest_categorical(
            "class_weight",
            [
                "none",
                "balanced",
            ],
        )

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        }   

        # GPU configuration (requires CUDA-enabled XGBoost build + compatible GPU)
        # NOTE: Current installed XGBoost build does not support GPU tree_method='gpu_hist'
        # on this environment (raises "Invalid Input: 'gpu_hist'"). We therefore
        # fall back to CPU 'hist'. To re-enable GPU, install a CUDA-enabled
        # XGBoost build and change tree_method / predictor / device accordingly.
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda",
            random_state=42,
            n_jobs=1,  # CV parallelism should be controlled externally if needed
            **params,
        )

        if scaler_choice == "none":
            scaler = None
        elif scaler_choice == "standard":
            scaler = StandardScaler()
        elif scaler_choice == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler choice: {scaler_choice}")

        if sampler_choice == "none":
            sampler = None
        elif sampler_choice == "random_over":
            sampler = RandomOverSampler(random_state=42)
        elif sampler_choice == "random_under":
            sampler = RandomUnderSampler(random_state=42)
        elif sampler_choice == "adasyn":
            sampler = ADASYN(random_state=42)
        elif sampler_choice == "smote":
            sampler = SMOTE(random_state=42)
        elif sampler_choice == "smote_tomek":
            sampler = SMOTETomek(random_state=42)
        elif sampler_choice == "smote_enn":
            sampler = SMOTEENN(random_state=42)
        else:
            raise ValueError(f"Unknown sampler choice: {sampler_choice}")

        pipe = fe_mod.build_pipeline(
            df_example=X,
            model=model,
            scaler=scaler,
            oversampler=sampler,
            config=cfg,
        )

        # Manual CV so we can pass fold-specific `sample_weight` when requested.
        fold_scores: list[float] = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            fit_kwargs = {}
            # Only use class weights when we are NOT applying any sampler.
            # Otherwise, the oversampler changes the effective sample sizes and
            # the precomputed weights based on y_train no longer match.
            use_class_weight = class_weight_choice == "balanced" and sampler_choice == "none"
            if use_class_weight:
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

sampler = TPESampler(seed=42)
pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)

study = optuna.create_study(
    direction="maximize",
    study_name=STUDY_NAME,
    storage=storage_url,
    load_if_exists=True,   # resume if job is requeued or restarted
    sampler=sampler,
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

best_scaler = best_params.get("scaler", "none")
best_sampler = best_params.get("sampler", "none")
best_class_weight = best_params.get("class_weight", "none")

if best_scaler == "none":
    final_scaler = None
elif best_scaler == "standard":
    final_scaler = StandardScaler()
elif best_scaler == "minmax":
    final_scaler = MinMaxScaler()
else:
    raise ValueError(f"Unknown scaler choice in best params: {best_scaler}")

if best_sampler == "none":
    final_sampler = None
elif best_sampler == "random_over":
    final_sampler = RandomOverSampler(random_state=42)
elif best_sampler == "random_under":
    final_sampler = RandomUnderSampler(random_state=42)
elif best_sampler == "adasyn":
    final_sampler = ADASYN(random_state=42)
elif best_sampler == "smote":
    final_sampler = SMOTE(random_state=42)
elif best_sampler == "smote_tomek":
    final_sampler = SMOTETomek(random_state=42)
elif best_sampler == "smote_enn":
    final_sampler = SMOTEENN(random_state=42)
else:
    raise ValueError(f"Unknown sampler choice in best params: {best_sampler}")

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
    df_example=X_train_full,
    model=final_model,
    scaler=final_scaler,
    oversampler=final_sampler,
    config=cfg,
)

fit_kwargs = {}
use_class_weight = best_class_weight == "balanced" and best_sampler == "none"
if use_class_weight:
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

