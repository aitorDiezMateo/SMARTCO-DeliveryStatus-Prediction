"""
Optuna hyperparameter tuning for AdaBoost with CV-safe feature engineering.

This is a lightweight alternative to XGBoost, suitable for CPUs without GPU support.
The AdaBoost classifier iteratively trains weak learners and focuses on 
misclassified samples.

Requirements:
  - optuna
  - scikit-learn
  - imbalanced-learn

This script loads `data/processed/train_raw.csv`, builds an sklearn Pipeline
with fold-safe encoders, and tunes AdaBoost hyperparameters using Stratified CV.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
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

TARGET_COL = "Delivery Status"


def main() -> None:
    df = pd.read_csv(TRAIN_RAW_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {TRAIN_RAW_PATH}")

    y_raw = df[TARGET_COL].astype(str)
    X = df.drop(columns=[TARGET_COL])

    # AdaBoost handles multiclass natively with SAMME algorithm
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = int(np.unique(y).size)

    fe_mod = _load_fe_module()
    cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna hyperparameter tuning."""
        
        # Hyperparameter suggestions for AdaBoost
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        learning_rate = trial.suggest_float("learning_rate", 0.5, 2.0)
        
        # Hyperparameters for the base DecisionTreeClassifier
        base_estimator_depth = trial.suggest_int("base_estimator_depth", 1, 5)
        base_estimator_min_samples_split = trial.suggest_int("base_estimator_min_samples_split", 2, 10)
        base_estimator_min_samples_leaf = trial.suggest_int("base_estimator_min_samples_leaf", 1, 5)

        # Create base estimator (shallow decision tree to keep it as a weak learner)
        base_estimator = DecisionTreeClassifier(
            max_depth=base_estimator_depth,
            min_samples_split=base_estimator_min_samples_split,
            min_samples_leaf=base_estimator_min_samples_leaf,
            random_state=42,
        )

        # Create AdaBoost classifier
        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
        )

        scaler = StandardScaler()
        sampler = RandomOverSampler(random_state=42)

        pipe = fe_mod.build_pipeline(
            df_example=X,
            model=model,
            scaler=scaler,
            oversampler=sampler,
            config=cfg,
        )

        # Optimize for macro F1 using cross-validation
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
        return float(scores.mean())

    # Create study with few trials for testing (use more runs in cluster)
    study = optuna.create_study(direction="maximize", study_name="AdaBoost")
    study.optimize(objective, n_trials=2, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("AdaBoost Hyperparameter Tuning Complete")
    print("=" * 60)
    print(f"Best CV f1_macro: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
