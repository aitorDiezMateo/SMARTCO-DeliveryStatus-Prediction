"""
This script runs Optuna hyperparameter tuning for:
  - a BaggingClassifier (DecisionTree base estimator), and
  - a RandomForestClassifier,
using CV-safe feature engineering and time-aware cross-validation.

It loads `data/processed/train_raw.csv`, builds an imblearn Pipeline with fold-safe
encoders, and tunes model hyperparameters using TimeSeriesSplit.
"""

import os
from pathlib import Path

import importlib.util
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


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
STUDY_NAME = "bagging"
N_TRIALS_PER_STUDY = 100
DB_PATH = OUTPUT_PATH / "bagging_study.db"
N_SPLITS = 3
RANDOM_STATE = 42


def _build_bagging(params: dict[str, object], *, oob_score: bool) -> BaggingClassifier:
    class_weight = None if params["estimator_class_weight"] == "none" else "balanced"

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
        oob_score=oob_score,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _build_random_forest(
    params: dict[str, object],
    *,
    oob_score: bool,
) -> RandomForestClassifier:
    class_weight = None if params["rf_class_weight"] == "none" else "balanced"
    bootstrap = bool(params["rf_bootstrap"])
    # sklearn raises when oob_score=True but bootstrap=False.
    effective_oob_score = bool(oob_score) and bootstrap

    return RandomForestClassifier(
        n_estimators=int(params["rf_n_estimators"]),
        max_depth=int(params["rf_max_depth"]),
        min_samples_split=int(params["rf_min_samples_split"]),
        min_samples_leaf=int(params["rf_min_samples_leaf"]),
        ccp_alpha=float(params["rf_ccp_alpha"]),
        class_weight=class_weight,
        max_features=float(params["rf_max_features"]),
        bootstrap=bootstrap,
        oob_score=effective_oob_score,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


######### Load training data #########

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
storage_url = f"sqlite:///{DB_PATH}"

print(f"Optuna storage: {storage_url}")
print(f"Study prefix:   {STUDY_NAME}")
print(f"N trials:       {N_TRIALS_PER_STUDY} per study (bagging + rf)")
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

def _common_trial_components(trial: optuna.Trial) -> tuple[float, str, object, str, str]:
    smoothing = trial.suggest_float("smoothing", 5.0, 100.0, log=True)
    sampler_choice = trial.suggest_categorical(
        "sampler",
        [
            "none",
            "random_over",
            "random_under",
            "adasyn",
            "smotenc",
            "smotenc_tomek",
            "smotenc_enn",
        ],
    )
    oversampler = fe_mod.make_resampler(
        sampler_choice,
        categorical_cols=list(cfg.low_card_cols) + list(cfg.high_card_cols),
        random_state=RANDOM_STATE,
    )

    # IMPORTANT (Optuna): categorical distributions must not change across trials within a study.
    # Always suggest from the full choice set, then override to "none" when resampling is enabled.
    estimator_class_weight = trial.suggest_categorical("estimator_class_weight", ["none", "balanced"])
    rf_class_weight = trial.suggest_categorical("rf_class_weight", ["none", "balanced"])
    if sampler_choice != "none":
        estimator_class_weight = "none"
        rf_class_weight = "none"

    return float(smoothing), str(sampler_choice), oversampler, str(estimator_class_weight), str(rf_class_weight)


def _cv_score(trial: optuna.Trial, *, model, oversampler, smoothing: float) -> float:
    pipe = fe_mod.build_pipeline(
        model=model,
        oversampler=oversampler,
        smoothing=float(smoothing),
        classes=list(range(n_classes)),
        config=cfg,
    )

    fold_scores: list[float] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        pipe.fit(X_train, y_train)
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


def objective_bagging(trial: optuna.Trial) -> float:
    smoothing, sampler_choice, oversampler, estimator_class_weight, _rf_class_weight = _common_trial_components(trial)
    params = {
        "model_type": "bagging",
        "sampler": sampler_choice,
        "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=50),
        "max_samples": trial.suggest_float("max_samples", 0.4, 1.0),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        "bootstrap_features": trial.suggest_categorical("bootstrap_features", [False, True]),
        "estimator_max_depth": trial.suggest_int("estimator_max_depth", 4, 40),
        "estimator_min_samples_split": trial.suggest_int("estimator_min_samples_split", 2, 30),
        "estimator_min_samples_leaf": trial.suggest_int("estimator_min_samples_leaf", 1, 15),
        "estimator_ccp_alpha": trial.suggest_float("estimator_ccp_alpha", 1e-6, 1e-2, log=True),
        "estimator_class_weight": estimator_class_weight,
        "smoothing": smoothing,
    }
    model = _build_bagging(params, oob_score=False)
    return _cv_score(trial, model=model, oversampler=oversampler, smoothing=smoothing)


def objective_rf(trial: optuna.Trial) -> float:
    smoothing, sampler_choice, oversampler, _estimator_class_weight, rf_class_weight = _common_trial_components(trial)
    params = {
        "model_type": "rf",
        "sampler": sampler_choice,
        "rf_n_estimators": trial.suggest_int("rf_n_estimators", 100, 900, step=50),
        "rf_max_depth": trial.suggest_int("rf_max_depth", 4, 40),
        "rf_min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 30),
        "rf_min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 15),
        "rf_ccp_alpha": trial.suggest_float("rf_ccp_alpha", 1e-6, 1e-2, log=True),
        "rf_class_weight": rf_class_weight,
        "rf_max_features": trial.suggest_float("rf_max_features", 0.5, 1.0),
        "rf_bootstrap": trial.suggest_categorical("rf_bootstrap", [True, False]),
        "smoothing": smoothing,
    }
    model = _build_random_forest(params, oob_score=False)
    return _cv_score(trial, model=model, oversampler=oversampler, smoothing=smoothing)


######### Create and run Optuna study #########

tpe_sampler = TPESampler(seed=RANDOM_STATE)
pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)

study_bagging = optuna.create_study(
    direction="maximize",
    study_name=f"{STUDY_NAME}__bagging",
    storage=storage_url,
    load_if_exists=True,
    sampler=tpe_sampler,
    pruner=pruner,
)
study_rf = optuna.create_study(
    direction="maximize",
    study_name=f"{STUDY_NAME}__rf",
    storage=storage_url,
    load_if_exists=True,
    sampler=tpe_sampler,
    pruner=pruner,
)

print("Running study:", study_bagging.study_name)
study_bagging.optimize(
    objective_bagging,
    n_trials=N_TRIALS_PER_STUDY,
    show_progress_bar=True,
    callbacks=[
        optuna.study.MaxTrialsCallback(N_TRIALS_PER_STUDY, states=(optuna.trial.TrialState.COMPLETE,)),
    ],
)

print("\nBagging best CV f1_macro:", study_bagging.best_value)
print("Bagging best params:")
for k, v in sorted(study_bagging.best_params.items()):
    print(f"  {k}: {v}")

pruned = len([t for t in study_bagging.trials if t.state == optuna.trial.TrialState.PRUNED])
complete = len([t for t in study_bagging.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f"Bagging trials: {complete} complete, {pruned} pruned")

print("\nRunning study:", study_rf.study_name)
study_rf.optimize(
    objective_rf,
    n_trials=N_TRIALS_PER_STUDY,
    show_progress_bar=True,
    callbacks=[
        optuna.study.MaxTrialsCallback(N_TRIALS_PER_STUDY, states=(optuna.trial.TrialState.COMPLETE,)),
    ],
)

print("\nRF best CV f1_macro:", study_rf.best_value)
print("RF best params:")
for k, v in sorted(study_rf.best_params.items()):
    print(f"  {k}: {v}")

pruned = len([t for t in study_rf.trials if t.state == optuna.trial.TrialState.PRUNED])
complete = len([t for t in study_rf.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f"RF trials: {complete} complete, {pruned} pruned")


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

def _train_eval(name: str, best_params: dict[str, object]) -> None:
    if name == "rf":
        final_model = _build_random_forest(best_params, oob_score=True)
    else:
        final_model = _build_bagging(best_params, oob_score=True)

    final_sampler = fe_mod.make_resampler(
        str(best_params.get("sampler", "none")),
        categorical_cols=list(cfg.low_card_cols) + list(cfg.high_card_cols),
        random_state=RANDOM_STATE,
    )
    final_pipe = fe_mod.build_pipeline(
        model=final_model,
        oversampler=final_sampler,
        smoothing=float(best_params.get("smoothing", 20.0)),
        classes=list(range(n_classes)),
        config=cfg,
    )

    final_pipe.fit(X_train_full, y_train_full)
    test_pred = final_pipe.predict(X_test)
    trained_model = final_pipe.named_steps["model"]

    print(f"\nTest evaluation ({name} best trial trained on full train):")
    print(f"  accuracy:          {accuracy_score(y_test, test_pred):.4f}")
    print(f"  balanced_accuracy: {balanced_accuracy_score(y_test, test_pred):.4f}")
    print(f"  f1_macro:          {f1_score(y_test, test_pred, average='macro'):.4f}")
    print(f"  f1_weighted:       {f1_score(y_test, test_pred, average='weighted'):.4f}")

    try:
        print(f"  oob_score:         {trained_model.oob_score_:.4f}")
    except Exception:  # noqa: BLE001
        print("  oob_score:         <not available>")

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


_train_eval("bagging", study_bagging.best_trial.params)
_train_eval("rf", study_rf.best_trial.params)