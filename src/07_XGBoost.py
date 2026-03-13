"""
Optuna hyperparameter tuning for GPU XGBoost with CV-safe feature engineering.

Requirements (install in your conda env):
  - optuna
  - xgboost
  - scikit-learn

This script loads `data/processed/train_raw.csv`, builds an sklearn Pipeline
with fold-safe encoders, and tunes XGBoost hyperparameters using Stratified CV.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
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

    # XGBoost needs numeric class labels for multiclass
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = int(np.unique(y).size)

    fe_mod = _load_fe_module()
    cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
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
        
        #TODO: Probar varios scalers
        scaler = StandardScaler()
        #TODO: Cambiar oversampler
        sampler = RandomOverSampler(random_state=42)

        pipe = fe_mod.build_pipeline(
            df_example=X,
            model=model,
            scaler=scaler,
            oversampler=sampler,
            config=cfg,
        )

        # Optimize for macro F1 by using sklearn's built-in scoring
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", study_name="XGBoost")
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    print("Best CV f1_macro:", study.best_value)
    print("Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

