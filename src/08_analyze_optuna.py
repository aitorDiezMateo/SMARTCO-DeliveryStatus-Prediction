"""
Analysis script for the Optuna XGBoost study.

Features:
- Connects to the same Optuna storage used in `07_XGBoost.py`.
- Builds a bar chart summarizing the best CV F1-macro per:
  - sampler method; and
  - for sampler == "none", by class_weight ("none"/"balanced").
- Trains a final pipeline with the best overall trial on the full train set
  and evaluates it on the test set, printing:
  - F1-macro
  - accuracy
  - recall-macro
  - ROC-AUC-macro (one-vs-rest, if possible)

NOTE: This script does NOT write anything to disk. Plots are shown interactively
with `plt.show()` and all metrics are printed to stdout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
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


DATA_PATH = Path(__file__).parent.parent / "data"
TRAIN_RAW_PATH = os.path.join(DATA_PATH, "processed", "train_raw.csv")
TEST_RAW_PATH = os.path.join(DATA_PATH, "processed", "test_raw.csv")
OUTPUT_PATH = Path(__file__).parent.parent / "output" / "optuna"

TARGET_COL = "Delivery Status"
DEFAULT_STUDY_NAME = "xgboost"
DEFAULT_DB = str(OUTPUT_PATH / "xgboost_study.db")


def _build_model_from_params(
    n_classes: int,
    params: dict,
) -> xgb.XGBClassifier:
    """Create an XGBClassifier from Optuna params (including tree params)."""
    tree_params = {
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "min_child_weight": params["min_child_weight"],
        "reg_lambda": params["reg_lambda"],
        "reg_alpha": params["reg_alpha"],
        "gamma": params["gamma"],
    }
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        random_state=42,
        n_jobs=1,
        **tree_params,
    )
    return model


def _make_scaler(choice: str):
    if choice == "none":
        return None
    if choice == "standard":
        return StandardScaler()
    if choice == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler choice: {choice}")


def _make_sampler(choice: str):
    if choice == "none":
        return None
    if choice == "random_over":
        return RandomOverSampler(random_state=42)
    if choice == "random_under":
        return RandomUnderSampler(random_state=42)
    if choice == "adasyn":
        return ADASYN(random_state=42)
    if choice == "smote":
        return SMOTE(random_state=42)
    if choice == "smote_tomek":
        return SMOTETomek(random_state=42)
    if choice == "smote_enn":
        return SMOTEENN(random_state=42)
    raise ValueError(f"Unknown sampler choice: {choice}")


def plot_sampler_results(trials: list[optuna.trial.FrozenTrial]) -> None:
    """Create a bar chart summarizing best F1 per sampler / class_weight."""
    records: list[dict] = []
    for t in trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        p = t.params
        value = float(t.value) if t.value is not None else np.nan
        records.append(
            {
                "sampler": p.get("sampler", "none"),
                "class_weight": p.get("class_weight", "none"),
                "value": value,
            }
        )

    if not records:
        print("No completed trials to plot.")
        return

    df = pd.DataFrame.from_records(records)

    # Best per sampler (aggregated over class_weight)
    best_by_sampler = df.groupby("sampler")["value"].max().reset_index()

    # Best for sampler == none, split by class_weight
    mask_none = df["sampler"] == "none"
    best_none = (
        df[mask_none]
        .groupby("class_weight")["value"]
        .max()
        .reset_index()
        .rename(columns={"class_weight": "sampler"})
    )
    best_none["sampler"] = best_none["sampler"].map(
        {"none": "none_cw_none", "balanced": "none_cw_balanced"}
    )

    plot_df = pd.concat(
        [
            best_by_sampler,
            best_none,
        ],
        ignore_index=True,
    )

    # Drop NaNs just in case
    plot_df = plot_df.dropna(subset=["value"])

    if plot_df.empty:
        print("No valid values to plot after filtering.")
        return

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["sampler"], plot_df["value"], color="#4472C4")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Best CV F1-macro")
    plt.title("Best CV F1-macro by sampler / class_weight (sampler=none)")
    plt.tight_layout()
    plt.show()


def evaluate_best_trial(
    study: optuna.Study,
) -> None:
    """Train best trial pipeline on full train and evaluate on test."""
    df_train = pd.read_csv(TRAIN_RAW_PATH)
    df_test = pd.read_csv(TEST_RAW_PATH)

    if TARGET_COL not in df_train.columns or TARGET_COL not in df_test.columns:
        raise ValueError(f"Target column '{TARGET_COL}' must be present in both train and test.")

    y_train_raw = df_train[TARGET_COL].astype(str)
    y_test_raw = df_test[TARGET_COL].astype(str)
    X_train = df_train.drop(columns=[TARGET_COL])
    X_test = df_test.drop(columns=[TARGET_COL])

    # Same label encoding convention as in 07_XGBoost.
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    n_classes = int(np.unique(y_train).size)

    best = study.best_trial
    params = best.params

    scaler_choice = params.get("scaler", "none")
    sampler_choice = params.get("sampler", "none")
    class_weight_choice = params.get("class_weight", "none")

    print("Best trial params:")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")
    print(f"  -> value (CV mean F1-macro): {best.value}")
    print()

    fe_mod = _load_fe_module()
    cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL)

    model = _build_model_from_params(n_classes=n_classes, params=params)
    scaler = _make_scaler(scaler_choice)
    sampler = _make_sampler(sampler_choice)

    pipe = fe_mod.build_pipeline(
        model=model,
        scaler=scaler,
        oversampler=sampler,
        config=cfg,
    )

    fit_kwargs = {}
    use_class_weight = class_weight_choice == "balanced" and sampler_choice == "none"
    if use_class_weight:
        fit_kwargs["model__sample_weight"] = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

    pipe.fit(X_train, y_train, **fit_kwargs)
    y_pred = pipe.predict(X_test)

    metrics = {
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "accuracy": accuracy_score(y_test, y_pred),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
    }

    # ROC-AUC macro (one-vs-rest) if we can get probabilities.
    try:
        y_proba = pipe.predict_proba(X_test)
        metrics["roc_auc_macro"] = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="macro",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Could not compute ROC-AUC (ignored): {exc}")

    print("Test metrics (best trial, trained on full train):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Optuna XGBoost study.")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help="Path or SQLAlchemy URL for the Optuna storage (default: SQLite in output/optuna/).",
    )
    parser.add_argument(
        "--study-name",
        default=DEFAULT_STUDY_NAME,
        help="Study name to load (default: %(default)s).",
    )
    args = parser.parse_args()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{args.db}" if not args.db.startswith("sqlite") else args.db
    print(f"Optuna storage: {storage_url}")
    print(f"Study name:     {args.study_name}")
    print()

    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage_url,
    )

    print(f"Loaded study with {len(study.trials)} trials.")
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  Complete trials: {complete}")
    print()

    # 1) Plot sampler / class_weight summary
    plot_sampler_results(study.trials)

    # 2) Train best trial on full train and evaluate on test
    evaluate_best_trial(study)


if __name__ == "__main__":
    main()

