"""
BASELINE MODELS - BENCHMARK COMPARISON

This script trains simple baseline models on RAW (non-engineered) data:
- LogisticRegression (statistical baseline)
- DummyClassifier (random/stratified baseline)

Purpose: Establish a performance floor to validate that complex models add value.

Inputs:  data/processed/train_raw.csv, test_raw.csv (raw, no feature engineering)
Outputs: output/predictions/logistic_regression_predictions.csv
         output/predictions/dummy_predictions.csv
         Metrics printed to stdout (no classification reports saved)

Always keep baseline predictions for comparison in ensemble analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data"
TRAIN_RAW_PATH = DATA_PATH / "processed" / "train_raw.csv"
TEST_RAW_PATH = DATA_PATH / "processed" / "test_raw.csv"

OUTPUT_PREDICTIONS = ROOT / "output" / "predictions"

TARGET_COL = "Delivery Status"
DATE_COL = "order date (DateOrders)"

RANDOM_STATE = 42


######### Load data #########

train_df = pd.read_csv(TRAIN_RAW_PATH)
test_df = pd.read_csv(TEST_RAW_PATH)

if TARGET_COL not in train_df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {TRAIN_RAW_PATH}")
if DATE_COL not in train_df.columns:
    raise ValueError(f"Date column '{DATE_COL}' not found in {TRAIN_RAW_PATH}")
if TARGET_COL not in test_df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {TEST_RAW_PATH}")
if DATE_COL not in test_df.columns:
    raise ValueError(f"Date column '{DATE_COL}' not found in {TEST_RAW_PATH}")

######### Encode target #########

le = LabelEncoder()

train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL], errors="coerce")
test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL], errors="coerce")

if train_df[DATE_COL].isna().any():
    raise ValueError("Train date parsing produced NaT values. Please check DATE_COL.")
if test_df[DATE_COL].isna().any():
    raise ValueError("Test date parsing produced NaT values. Please check DATE_COL.")

train_df["order_year"] = train_df[DATE_COL].dt.year
train_df["order_month"] = train_df[DATE_COL].dt.month
train_df["order_day"] = train_df[DATE_COL].dt.day
train_df["day_of_week"] = train_df[DATE_COL].dt.dayofweek
train_df["is_weekend"] = train_df[DATE_COL].dt.weekday.isin([5, 6]).astype(int)

test_df["order_year"] = test_df[DATE_COL].dt.year
test_df["order_month"] = test_df[DATE_COL].dt.month
test_df["order_day"] = test_df[DATE_COL].dt.day
test_df["day_of_week"] = test_df[DATE_COL].dt.dayofweek
test_df["is_weekend"] = test_df[DATE_COL].dt.weekday.isin([5, 6]).astype(int)

y_train_raw = train_df[TARGET_COL].astype(str)
y_test_raw = test_df[TARGET_COL].astype(str)

y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)

######### Prepare features (raw, but make them ML-ready) #########

# Keep it minimal: we only convert the date to numeric parts and one-hot encode categoricals.
train_df = train_df.drop(columns=[DATE_COL])
test_df = test_df.drop(columns=[DATE_COL])

X_train = train_df.drop(columns=[TARGET_COL])
X_test = test_df.drop(columns=[TARGET_COL])

numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

######### LogisticRegression #########

lr = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
    multi_class="auto",
)

lr_model = Pipeline(steps=[("preprocess", preprocess), ("model", lr)])

print("=== Training LogisticRegression ===")
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

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

y_proba: Optional[np.ndarray]
try:
    y_proba = lr_model.predict_proba(X_test)
except Exception:  # noqa: BLE001
    y_proba = None

if y_proba is not None:
    print("=== Probabilistic metrics (one-vs-rest for multiclass) ===")
    print(f"ROC-AUC macro (OVR): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'):.6f}")
    print(f"ROC-AUC weighted (OVR): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'):.6f}")

    n_classes = int(np.unique(y_test).size)
    y_true_ovr = np.eye(n_classes, dtype=int)[y_test]
    pr_auc_macro = average_precision_score(y_true_ovr, y_proba, average="macro")
    pr_auc_weighted = average_precision_score(y_true_ovr, y_proba, average="weighted")
    print(f"PR-AUC macro (OVR): {pr_auc_macro:.6f}")
    print(f"PR-AUC weighted (OVR): {pr_auc_weighted:.6f}")

OUTPUT_PREDICTIONS.mkdir(parents=True, exist_ok=True)

y_pred_labels = le.inverse_transform(y_pred)
preds_df = test_df[[TARGET_COL]].copy()
preds_df["prediction"] = y_pred_labels

if y_proba is not None:
    for class_idx, class_name in enumerate(le.classes_):
        preds_df[f"proba_{class_name}"] = y_proba[:, class_idx]

pred_path = OUTPUT_PREDICTIONS / "logistic_regression_predictions.csv"
preds_df.to_csv(pred_path, index=False)
print("\nSaved predictions:")
print(f"- {pred_path}")

######### DummyClassifier #########

dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
X_train_dummy = np.zeros((X_train.shape[0], 1), dtype=float)
X_test_dummy = np.zeros((X_test.shape[0], 1), dtype=float)

print("\n=== Training DummyClassifier ===")
dummy.fit(X_train_dummy, y_train)

y_pred = dummy.predict(X_test_dummy)

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

y_proba = None
try:
    y_proba = dummy.predict_proba(X_test_dummy)
except Exception:  # noqa: BLE001
    y_proba = None

if y_proba is not None:
    print("=== Probabilistic metrics (one-vs-rest for multiclass) ===")
    print(f"ROC-AUC macro (OVR): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'):.6f}")
    print(f"ROC-AUC weighted (OVR): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'):.6f}")

    n_classes = int(np.unique(y_test).size)
    y_true_ovr = np.eye(n_classes, dtype=int)[y_test]
    pr_auc_macro = average_precision_score(y_true_ovr, y_proba, average="macro")
    pr_auc_weighted = average_precision_score(y_true_ovr, y_proba, average="weighted")
    print(f"PR-AUC macro (OVR): {pr_auc_macro:.6f}")
    print(f"PR-AUC weighted (OVR): {pr_auc_weighted:.6f}")

y_pred_labels = le.inverse_transform(y_pred)
preds_df = test_df[[TARGET_COL]].copy()
preds_df["prediction"] = y_pred_labels

if y_proba is not None:
    for class_idx, class_name in enumerate(le.classes_):
        preds_df[f"proba_{class_name}"] = y_proba[:, class_idx]

pred_path = OUTPUT_PREDICTIONS / "dummy_predictions.csv"
preds_df.to_csv(pred_path, index=False)
print("\nSaved predictions:")
print(f"- {pred_path}")

