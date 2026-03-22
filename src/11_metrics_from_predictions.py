"""
METRICS AGGREGATION - ALL MODELS

This script aggregates evaluation metrics across all model predictions:
- Loads all saved predictions from output/predictions/*.csv
- Computes per-model metrics:  accuracy, balanced accuracy, F1, precision, recall
- Computes probability-based metrics (if proba_* columns exist): ROC-AUC, PR-AUC (one-vs-rest)
- Generates confusion matrices

Inputs:  output/predictions/*.csv (from 09_best_models_individual.py, 08_Voting.py, 10_benchmark.py)
Outputs: output/metrics/all_models_metrics.csv (summary table for comparison)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
)

ROOT = Path(__file__).parent.parent
PREDICTIONS_DIR = ROOT / "output" / "predictions"
OUTPUT_METRICS = ROOT / "output" / "metrics"

TARGET_COL = "Delivery Status"
PRED_COL = "prediction"

######### Load prediction files #########

if not PREDICTIONS_DIR.exists():
    raise FileNotFoundError(f"Predictions directory not found: {PREDICTIONS_DIR}")

pred_files = sorted(PREDICTIONS_DIR.glob("*.csv"))
if not pred_files:
    raise FileNotFoundError(f"No prediction CSVs found in: {PREDICTIONS_DIR}")


rows: list[dict[str, object]] = []

######### Compute metrics per model #########

for pred_path in pred_files:
    if pred_path.name == "voting_hard_predictions.csv":
        model_name = "Hard Voting"
    elif pred_path.name == "voting_soft_predictions.csv":
        model_name = "Soft Voting"
    else:
        model_name = pred_path.stem
        if model_name.endswith("_predictions"):
            model_name = model_name[: -len("_predictions")]
    df = pd.read_csv(pred_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{pred_path.name}: missing '{TARGET_COL}' column")
    if PRED_COL not in df.columns:
        raise ValueError(f"{pred_path.name}: missing '{PRED_COL}' column")

    y_true = df[TARGET_COL].astype(str).to_numpy()
    y_pred = df[PRED_COL].astype(str).to_numpy()

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    macro = report["macro avg"]
    weighted = report["weighted avg"]

    row: dict[str, object] = {
        "model": model_name,
        "accuracy": float(report["accuracy"]),
        "precision_macro": float(macro["precision"]),
        "recall_macro": float(macro["recall"]),
        "f1_macro": float(macro["f1-score"]),
        "precision_weighted": float(weighted["precision"]),
        "recall_weighted": float(weighted["recall"]),
        "f1_weighted": float(weighted["f1-score"]),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    proba_cols = [c for c in df.columns if c.startswith("proba_")]
    if proba_cols:
        # Preserve column order as stored by the producing script.
        classes_order = [c[len("proba_") :] for c in proba_cols]
        class_to_idx = {cls: i for i, cls in enumerate(classes_order)}

        if not set(y_true).issubset(set(classes_order)):
            missing = sorted(set(y_true) - set(classes_order))
            raise ValueError(f"{pred_path.name}: proba_* missing classes: {missing}")

        y_true_idx = np.array([class_to_idx[v] for v in y_true], dtype=int)
        y_proba = df[proba_cols].to_numpy(dtype=float)

        row["roc_auc_macro_ovr"] = float(
            roc_auc_score(y_true_idx, y_proba, multi_class="ovr", average="macro")
        )
        row["roc_auc_weighted_ovr"] = float(
            roc_auc_score(y_true_idx, y_proba, multi_class="ovr", average="weighted")
        )

        n_classes = int(y_proba.shape[1])
        y_true_ovr = np.eye(n_classes, dtype=int)[y_true_idx]
        row["pr_auc_macro_ovr"] = float(average_precision_score(y_true_ovr, y_proba, average="macro"))
        row["pr_auc_weighted_ovr"] = float(
            average_precision_score(y_true_ovr, y_proba, average="weighted")
        )

    rows.append(row)


######### Save metrics #########

OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT_METRICS / "all_models_metrics.csv"

metrics_df = pd.DataFrame(rows).round(4)
metrics_df = metrics_df.sort_values(["f1_macro"], ascending=False)
metrics_df.to_csv(out_path, index=False)

print("=== Saved metrics summary ===")
print(f"- {out_path}")

