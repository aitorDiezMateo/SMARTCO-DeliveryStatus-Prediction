"""
Train a VotingClassifier for Delivery Status prediction using the best Optuna hyperparameters from:

- XGBoost (`output/optuna/xgboost_study.db`, study `xgboost_study`)
- CatBoost (`output/optuna/catboost_study.db`, study `catboost_v2`)
- Bagging (`output/optuna/bagging_study.db`, study `bagging__bagging`)

The script trains on `data/processed/train_raw.csv`, evaluates on
`data/processed/test_raw.csv`, prints metrics, and saves:

- `data/processed/voting_classification_report.csv`
- `data/processed/voting_predictions.csv`
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
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.metrics import (
	average_precision_score,
	balanced_accuracy_score,
	classification_report,
	f1_score,
	matthews_corrcoef,
	roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

######### Paths and constants #########

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data"
TRAIN_RAW_PATH = DATA_PATH / "processed" / "train_raw.csv"
TEST_RAW_PATH = DATA_PATH / "processed" / "test_raw.csv"
OUTPUT_PROCESSED = DATA_PATH / "processed"
OUTPUT_OPTUNA = ROOT / "output" / "optuna"

TARGET_COL = "Delivery Status"
DATE_COL = "order date (DateOrders)"
RANDOM_STATE = 42

# Voting configuration: set VOTING_MODE=hard or VOTING_MODE=soft.
VOTING_MODE = os.environ.get("VOTING_MODE", "soft").strip().lower()
if VOTING_MODE not in {"hard", "soft"}:
	raise ValueError("VOTING_MODE must be 'hard' or 'soft'.")


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
		try:
			available = [s.study_name for s in optuna.get_all_study_summaries(storage=storage_url)]
		except Exception:  # noqa: BLE001
			available = []
		raise KeyError(
			f"Optuna study '{study_name}' not found in DB: {db_path}. "
			f"Available studies: {available}."
		) from exc
	return dict(study.best_trial.params)


def build_best_xgboost_pipeline():
	fe_mod = _load_fe_module()
	cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL, date_col=DATE_COL)

	db_path = OUTPUT_OPTUNA / "xgboost_study.db"
	best = _load_best_params(db_path=db_path, study_name="xgboost")

	sampler_choice = str(best.get("sampler", "none"))
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
        "db_path": db_path,
        "study_name": "xgboost_study",
		"sampler": sampler_choice,
        "class_weight": str(best.get("estimator_class_weight", "none")),
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


def build_best_bagging_pipeline():
	fe_mod = _load_fe_module()
	cfg = fe_mod.FeatureEngineeringConfig(target_col=TARGET_COL, date_col=DATE_COL)

	db_path = OUTPUT_OPTUNA / "bagging_study.db"
	best = _load_best_params(db_path=db_path, study_name="bagging")

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
        "db_path": db_path,
        "study_name": "bagging__bagging",
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


class CatBoostAutoCatFeatures(ClassifierMixin, BaseEstimator):
	"""Sklearn-compatible wrapper that auto-infers CatBoost categorical indices."""

	def __init__(self, model: Any, *, prep_cfg: CatBoostPrepConfig | None = None) -> None:
		self.model = model
		self.prep_cfg = prep_cfg or CatBoostPrepConfig()

	def fit(self, X, y):  # noqa: ANN001
		X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
		X_prep, cat_idx = _catboost_prepare_features(X_df, self.prep_cfg)
		self.model.fit(X_prep, y, cat_features=cat_idx)
		self.classes_ = np.unique(y)
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
		from catboost import CatBoostClassifier
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"catboost is not installed in the current environment. "
			"Install it (e.g. `pip install catboost`) to build the CatBoost model."
		) from exc

	db_path = OUTPUT_OPTUNA / "catboost_study.db"
	best = _load_best_params(db_path=db_path, study_name="catboost_v2")

	task_type = os.environ.get("CATBOOST_TASK_TYPE", "GPU").upper()
	devices = os.environ.get("CATBOOST_DEVICES", "0")

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
	return {
		"model": wrapped,
		"raw_model": model,
		"best_params": best,
		"sampler": sampler_choice,
		"class_weight_method": class_weight_method
	}


######### Load data #########

train_df = pd.read_csv(TRAIN_RAW_PATH)
test_df = pd.read_csv(TEST_RAW_PATH)

for split_name, split_df, split_path in [
	("train", train_df, TRAIN_RAW_PATH),
	("test", test_df, TEST_RAW_PATH),
]:
	if TARGET_COL not in split_df.columns:
		raise ValueError(f"Target column '{TARGET_COL}' not found in {split_path}")
	if DATE_COL not in split_df.columns:
		raise ValueError(f"Date column '{DATE_COL}' not found in {split_path}")

	split_df[DATE_COL] = pd.to_datetime(split_df[DATE_COL], errors="coerce")
	if split_df[DATE_COL].isna().any():
		raise ValueError(
			f"Date column '{DATE_COL}' contains invalid values in {split_name}. "
			"Please ensure it can be parsed as datetime."
		)

train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)
test_df = test_df.sort_values(DATE_COL).reset_index(drop=True)

X_train = train_df.drop(columns=[TARGET_COL])
X_test = test_df.drop(columns=[TARGET_COL])
y_train_raw = train_df[TARGET_COL].astype(str)
y_test_raw = test_df[TARGET_COL].astype(str)

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

######### Build best base estimators #########

xgb_bundle = build_best_xgboost_pipeline()
cat_bundle = build_best_catboost_model()
bag_bundle = build_best_bagging_pipeline()

estimators = [
	("xgboost", xgb_bundle["pipeline"]),
	("catboost", cat_bundle["model"]),
	("bagging", bag_bundle["pipeline"]),
]

voting_clf = VotingClassifier(
	estimators=estimators,
	voting=VOTING_MODE,
	# Equal contribution from all 3 tuned models.
	weights=[1.0, 1.0, 1.0],
	n_jobs=1,
)

######### Train #########

print("=== Training VotingClassifier ===")
print(f"voting={VOTING_MODE}")
voting_clf.fit(X_train, y_train)

######### Evaluate #########

y_pred = voting_clf.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)

if VOTING_MODE == "soft":
	y_proba = voting_clf.predict_proba(X_test)
else:
	y_proba = None

print("=== Classification report (macro/weighted are imbalance-friendly) ===")
report_text = classification_report(
	y_test,
	y_pred,
	target_names=[str(c) for c in le.classes_],
	digits=4,
	zero_division=0,
)
print(report_text)

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

######### Save outputs #########

OUTPUT_PROCESSED.mkdir(parents=True, exist_ok=True)

report_dict = classification_report(
	y_test,
	y_pred,
	target_names=[str(c) for c in le.classes_],
	output_dict=True,
	zero_division=0,
)
report_df = pd.DataFrame(report_dict).transpose()
report_path = OUTPUT_PROCESSED / "voting_classification_report.csv"
report_df.to_csv(report_path, index=True)

preds_df = test_df[[TARGET_COL]].copy()
preds_df["prediction"] = y_pred_labels

if y_proba is not None:
	for class_idx, class_name in enumerate(le.classes_):
		preds_df[f"proba_{class_name}"] = y_proba[:, class_idx]

pred_path = OUTPUT_PROCESSED / "voting_predictions.csv"
preds_df.to_csv(pred_path, index=False)

print("\nSaved files:")
print(f"- {report_path}")
print(f"- {pred_path}")
