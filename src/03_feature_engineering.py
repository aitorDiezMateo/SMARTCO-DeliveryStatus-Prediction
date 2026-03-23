"""
PRODUCTION PIPELINE DOCUMENTATION - FEATURE ENGINEERING REFERENCE

This script documents and visualizes ONLY transformations that are actually used
in the real training pipeline (`src/pipelines/feature_engineering.py`).

What this script includes (used in production):
- Deterministic date-based features (order_year, order_month, order_day,
  is_weekend, day_of_week, week_of_year, quarter, is_month_start, is_month_end)
- Cross-border flag (is_cross_border)
- Shipping route feature (shipping_route)
- Frequency encoding for high-cardinality columns

What this script intentionally excludes (not materialized here):
- HistoricalTargetStats past-only target features (computed fold-safely inside CV)
- OneHotEncoder / TargetMeanEncoder outputs from the final ColumnTransformer
- Any exploratory-only transformations not used in production

Use this script for methodology documentation and feature-distribution plots.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path(__file__).parent.parent / "data"
TRAIN_RAW_PATH = os.path.join(DATA_PATH, "processed", "train_raw.csv")
TEST_RAW_PATH = os.path.join(DATA_PATH, "processed", "test_raw.csv")

DATE_COL = "order date (DateOrders)"

FREQ_ENCODE_COLS = [
    "Order City",
    "Customer City",
    "Product Name",
    "Category Name",
    "Order Region",
    "Market",
]

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "feature_engineering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FE_DATA_PATH = DATA_PATH / "feature_engineering"
FE_DATA_PATH.mkdir(parents=True, exist_ok=True)

######### Load processed datasets #########
train_df = pd.read_csv(TRAIN_RAW_PATH)
test_df = pd.read_csv(TEST_RAW_PATH)

######### Validate required columns #########
required_cols = [
    DATE_COL,
    "Customer Country",
    "Order Country",
    "Order Region",
    "Market",
    *FREQ_ENCODE_COLS,
]
missing_train = [c for c in required_cols if c not in train_df.columns]
missing_test = [c for c in required_cols if c not in test_df.columns]
if missing_train:
    raise ValueError("Missing required columns in train_df: " + ", ".join(missing_train))
if missing_test:
    raise ValueError("Missing required columns in test_df: " + ", ".join(missing_test))

######### Deterministic features (mirror FeatureBuilder in pipeline) #########
def _apply_feature_builder_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    dates = pd.to_datetime(out[DATE_COL], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Column '{DATE_COL}' contains invalid datetime values.")

    out["order_year"] = dates.dt.year
    out["order_month"] = dates.dt.month
    out["order_day"] = dates.dt.day
    out["is_weekend"] = dates.dt.weekday.isin([5, 6]).astype(int)
    out["day_of_week"] = dates.dt.dayofweek
    out["week_of_year"] = dates.dt.isocalendar().week.values.astype(int)
    out["quarter"] = dates.dt.quarter
    out["is_month_start"] = dates.dt.is_month_start.astype(int)
    out["is_month_end"] = dates.dt.is_month_end.astype(int)

    out["is_cross_border"] = (out["Customer Country"] != out["Order Country"]).astype(int)

    origin = out["Order Region"].astype(str).str.replace(" ", "_", regex=False)
    destination = out["Market"].astype(str).str.replace(" ", "_", regex=False)
    out["shipping_route"] = origin + "_to_" + destination

    return out


train_fe = _apply_feature_builder_like(train_df)
test_fe = _apply_feature_builder_like(test_df)

######### Frequency encoding (train-fitted map, test transform) #########
for col in FREQ_ENCODE_COLS:
    counts = train_fe[col].value_counts()
    train_fe[f"{col}_freq"] = train_fe[col].map(counts).fillna(1).astype(int)
    test_fe[f"{col}_freq"] = test_fe[col].map(counts).fillna(1).astype(int)

######### Production note #########
print(
    "NOTE: HistoricalTargetStats and final encoded matrices (OneHot/TargetMean) "
    "are generated inside src/pipelines/feature_engineering.py during CV/training."
)

######### Plot deterministic engineered features (discrete count plots) #########
sns.set_theme(style="whitegrid")

ENGINEERED_DISCRETE = [
    "order_year",
    "order_month",
    "order_day",
    "is_weekend",
    "day_of_week",
    "week_of_year",
    "quarter",
    "is_month_start",
    "is_month_end",
    "is_cross_border",
]

for col in ENGINEERED_DISCRETE:
    if col not in train_fe.columns or col not in test_fe.columns:
        raise ValueError(f"Expected engineered column '{col}' not found in train/test outputs.")


def _plot_discrete_count_grid(df: pd.DataFrame, columns: list[str], title: str, out_path: Path) -> None:
    n_features = len(columns)
    if n_features == 0:
        return

    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3.8 * nrows), constrained_layout=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        x = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        counts = x.value_counts().sort_index()

        ax.bar(
            counts.index.astype(str),
            counts.values,
            color="#4C78A8",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )

        # Avoid unreadable x-axis labels for high-cardinality discrete features.
        if len(counts) > 14:
            for tick_idx, label in enumerate(ax.get_xticklabels()):
                if tick_idx % 2 == 1:
                    label.set_visible(False)

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    fig.savefig(out_path, format="svg", dpi=200)
    plt.close(fig)


def _plot_frequency_hist_grid(df: pd.DataFrame, columns: list[str], title: str, out_path: Path) -> None:
    n_features = len(columns)
    if n_features == 0:
        return

    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3.8 * nrows), constrained_layout=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        x = pd.to_numeric(df[col], errors="coerce").dropna()

        ax.hist(
            x,
            bins=30,
            color="#2A9D8F",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Frequency value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    fig.savefig(out_path, format="svg", dpi=200)
    plt.close(fig)


# If a deterministic feature is constant in train, it adds little value as a plot.
plot_discrete_cols = [c for c in ENGINEERED_DISCRETE if train_fe[c].nunique(dropna=True) > 1]
skipped_constant_cols = [c for c in ENGINEERED_DISCRETE if c not in plot_discrete_cols]
if skipped_constant_cols:
    print("Skipping constant deterministic features in train: " + ", ".join(skipped_constant_cols))


_plot_discrete_count_grid(
    train_fe,
    plot_discrete_cols,
    "Production deterministic engineered features (train)",
    OUTPUT_DIR / "01_production_engineered_numeric_train.svg",
)

_plot_discrete_count_grid(
    test_fe,
    plot_discrete_cols,
    "Production deterministic engineered features (test)",
    OUTPUT_DIR / "01_production_engineered_numeric_test.svg",
)

######### Plot frequency-encoded feature distributions #########
FREQ_COLS_OUTPUT = [f"{c}_freq" for c in FREQ_ENCODE_COLS]

_plot_frequency_hist_grid(
    train_fe,
    FREQ_COLS_OUTPUT,
    "Production frequency-encoded features (train)",
    OUTPUT_DIR / "02_production_frequency_features_train.svg",
)

_plot_frequency_hist_grid(
    test_fe,
    FREQ_COLS_OUTPUT,
    "Production frequency-encoded features (test)",
    OUTPUT_DIR / "02_production_frequency_features_test.svg",
)

######### Save documentation datasets #########
train_fe_path = FE_DATA_PATH / "train_features_production_doc.csv"
test_fe_path = FE_DATA_PATH / "test_features_production_doc.csv"

train_fe.to_csv(train_fe_path, index=False)
test_fe.to_csv(test_fe_path, index=False)

print(f"Saved documented production features (train): {train_fe_path}")
print(f"Saved documented production features (test):  {test_fe_path}")
