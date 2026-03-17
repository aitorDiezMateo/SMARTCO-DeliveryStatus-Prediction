"""
Optuna hyperparameter tuning and final training for a Bagging classifier.

This script tunes a BaggingClassifier built on top of a DecisionTreeClassifier,
using the project dataset and the shared feature-engineering pipeline. After the
study finishes, it retrains the best configuration on the full training split
and saves evaluation artifacts for the holdout test split.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent.parent / "data"
TRAIN_RAW_PATH = os.path.join(DATA_PATH, "processed", "train_raw.csv")
TEST_RAW_PATH = os.path.join(DATA_PATH, "processed", "test_raw.csv")
DATE_COL = "order date (DateOrders)"

TARGET_COL = "Delivery Status"
RANDOM_STATE = 42
CV_SPLITS = 5
N_TRIALS = int(os.environ.get("OPTUNA_N_TRIALS", "30"))
OPTUNA_SCORING = "f1_macro"
BEST_PARAMS_CSV = DATA_PATH / "processed" / "bagging_best_params.csv"


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


def _metric_summary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
	"""Compute the core classification metrics used across the project."""
	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
		"macro_f1": f1_score(y_true, y_pred, average="macro"),
		"weighted_f1": f1_score(y_true, y_pred, average="weighted"),
	}


def _build_bagging_classifier(params: dict[str, object], *, oob_score: bool) -> BaggingClassifier:
	"""Create a BaggingClassifier from Optuna parameters."""
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


def _build_objective(X: pd.DataFrame, y: pd.Series, fe_mod, cfg, cv: TimeSeriesSplit):
	"""Create the Optuna objective with the project feature-engineering pipeline."""

	def objective(trial: optuna.Trial) -> float:
		params = {
			"n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
			"max_samples": trial.suggest_float("max_samples", 0.4, 1.0),
			"max_features": trial.suggest_float("max_features", 0.5, 1.0),
			"bootstrap_features": trial.suggest_categorical("bootstrap_features", [False, True]),
			"estimator_max_depth": trial.suggest_int("estimator_max_depth", 4, 40),
			"estimator_min_samples_split": trial.suggest_int("estimator_min_samples_split", 2, 30),
			"estimator_min_samples_leaf": trial.suggest_int("estimator_min_samples_leaf", 1, 15),
			"estimator_ccp_alpha": trial.suggest_float("estimator_ccp_alpha", 1e-6, 1e-2, log=True),
			"estimator_class_weight": trial.suggest_categorical(
				"estimator_class_weight",
				["none", "balanced"],
			),
		}

		model = _build_bagging_classifier(params, oob_score=False)
		pipeline = fe_mod.build_pipeline(
			df_example=X,
			model=model,
			scaler=None,
			oversampler=None,
			config=cfg,
		)

		scores = cross_val_score(
			pipeline,
			X,
			y,
			cv=cv,
			scoring=OPTUNA_SCORING,
			n_jobs=1,
			error_score="raise",
		)
		return float(scores.mean())

	return objective

def _save_best_params_csv(study: optuna.Study, out_path: Path) -> None:
	rows = [{"param": k, "value": v} for k, v in sorted(study.best_params.items())]
	out_path.parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(rows).to_csv(out_path, index=False)

def main() -> None:
	train_df = pd.read_csv(TRAIN_RAW_PATH)
	test_df = pd.read_csv(TEST_RAW_PATH)

	if TARGET_COL not in train_df.columns:
		raise ValueError(f"Target column '{TARGET_COL}' not found in {TRAIN_RAW_PATH}")
	if TARGET_COL not in test_df.columns:
		raise ValueError(f"Target column '{TARGET_COL}' not found in {TEST_RAW_PATH}")

	X_train = train_df.drop(columns=[TARGET_COL])
	y_train = train_df[TARGET_COL]

	if DATE_COL not in X_train.columns:
		raise ValueError(f"Date column '{DATE_COL}' not found in training data.")
	order = pd.to_datetime(X_train[DATE_COL], errors="coerce").sort_values().index
	X_train = X_train.loc[order].reset_index(drop=True)
	y_train = y_train.loc[order].reset_index(drop=True)

	X_test = test_df.drop(columns=[TARGET_COL])
	y_test = test_df[TARGET_COL]

	fe_mod = _load_fe_module()
	cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL)
	cv = TimeSeriesSplit(n_splits=CV_SPLITS)
	study = optuna.create_study(
		direction="maximize",
		study_name="BaggingClassifier",
		pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
		sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
	)
	objective = _build_objective(X_train, y_train, fe_mod, cfg, cv)
	study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
	_save_best_params_csv(study, BEST_PARAMS_CSV)

	bagging_model = _build_bagging_classifier(study.best_params, oob_score=True)

	pipeline = fe_mod.build_pipeline(
		df_example=X_train,
		model=bagging_model,
		scaler=None,
		oversampler=None,
		config=cfg,
	)

	pipeline.fit(X_train, y_train)

	trained_bagging = pipeline.named_steps["model"]
	y_train_pred = pipeline.predict(X_train)
	y_test_pred = pipeline.predict(X_test)

	train_metrics = _metric_summary(y_train, y_train_pred)
	test_metrics = _metric_summary(y_test, y_test_pred)

	print("Bagging Model: DecisionTreeClassifier ensemble")
	print(f"Optuna trials: {N_TRIALS}")
	print(f"Best CV {OPTUNA_SCORING}: {study.best_value:.4f}")
	print("Best hyperparameters:")
	for key, value in sorted(study.best_params.items()):
		print(f"  {key}: {value}")
	print(f"OOB score: {trained_bagging.oob_score_:.4f}")
	print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
	print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
	print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
	print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
	print(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
	print("\nClassification Report:")
	print(classification_report(y_test, y_test_pred))


if __name__ == "__main__":
	main()
