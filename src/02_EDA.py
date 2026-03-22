"""
EXPLORATORY DATA ANALYSIS - STAGE 2

This script performs comprehensive EDA on the cleaned dataset:
- Generates automated profiling report with ydata-profiling
- Visualizes distributions of numerical and categorical features
- Compares train vs test set characteristics for drift detection

Dependency: Requires data/processed/ outputs from 01_prepare_data.py
Inputs:  data/processed/train_raw.csv, test_raw.csv
Outputs: output/eda/03_ydata_profile_train.html (profiling report)

This is an EXPLORATORY STAGE (not part of model training pipeline).
Use for methodology documentation and data quality assessment.
"""

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport


DATA_PATH = Path(__file__).parent.parent / "data"
TRAIN_RAW_PATH = os.path.join(DATA_PATH, "processed", "train_raw.csv")
TEST_RAW_PATH = os.path.join(DATA_PATH, "processed", "test_raw.csv")

######### Load processed datasets #########
train_df = pd.read_csv(TRAIN_RAW_PATH)
test_df = pd.read_csv(TEST_RAW_PATH)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

DATE_COL = "order date (DateOrders)"

######### Numerical features configuration #########
# For EDA we want to cover all numerical variables in the dataset, except:
# - The date column (handled separately as a time series)
# - "Order Item Quantity", which we treat as categorical for EDA

all_numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()

NUMERICAL_FEATURES = [
    c for c in all_numeric_cols if c not in {"Order Item Quantity"}
]

numerical_train_df = train_df[NUMERICAL_FEATURES].copy()
numerical_test_df = test_df[NUMERICAL_FEATURES].copy()

for c in NUMERICAL_FEATURES:
    numerical_train_df[c] = pd.to_numeric(numerical_train_df[c], errors="coerce")
    numerical_test_df[c] = pd.to_numeric(numerical_test_df[c], errors="coerce")

######### Plot numerical distributions (train) #########
OUT_FIG_NUM_TRAIN = OUTPUT_DIR / "00_numerical_distribution.svg"

n_features = len(NUMERICAL_FEATURES)
ncols = 3
nrows = math.ceil(n_features / ncols)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(5.0 * ncols, 3.5 * nrows),
)
axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

palette_color = "#4472C4"

for i, col in enumerate(NUMERICAL_FEATURES):
    ax = axes[i]
    x = numerical_train_df[col].dropna()

    ax.hist(
        x,
        bins=40,
        color=palette_color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
        zorder=1,
    )

    if x.nunique() > 1:
        ax2 = ax.twinx()
        sns.kdeplot(
            x=x,
            ax=ax2,
            color="#D62728",
            linewidth=2.0,
            fill=False,
            zorder=3,
        )
        ax2.set_ylabel("")
        ax2.set_yticks([])
        ax2.tick_params(axis="y", length=0)
        ax2.grid(False)
        ax2.patch.set_alpha(0)
        ax2.spines["right"].set_visible(False)

    ax.set_title(col, fontsize=11, pad=10)
    ax.set_xlabel("Value", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

for j in range(n_features, len(axes)):
    axes[j].axis("off")

fig.suptitle("Numerical feature distributions (train set)", fontsize=16)
fig.tight_layout(rect=[0, 0.0, 1, 0.96])
fig.savefig(OUT_FIG_NUM_TRAIN, format="svg", dpi=200)
plt.close(fig)

######### Plot numerical distributions (test) #########
OUT_FIG_NUM_TEST = OUTPUT_DIR / "00_numerical_distribution_test.svg"

n_features_test = len(NUMERICAL_FEATURES)
ncols_test = 3
nrows_test = math.ceil(n_features_test / ncols_test)

fig, axes = plt.subplots(
    nrows=nrows_test,
    ncols=ncols_test,
    figsize=(5.0 * ncols_test, 3.5 * nrows_test),
)
axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, col in enumerate(NUMERICAL_FEATURES):
    ax = axes[i]
    x = numerical_test_df[col].dropna()

    ax.hist(
        x,
        bins=40,
        color=palette_color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
        zorder=1,
    )

    if x.nunique() > 1:
        ax2 = ax.twinx()
        sns.kdeplot(
            x=x,
            ax=ax2,
            color="#D62728",
            linewidth=2.0,
            fill=False,
            zorder=3,
        )
        ax2.set_ylabel("")
        ax2.set_yticks([])
        ax2.tick_params(axis="y", length=0)
        ax2.grid(False)
        ax2.patch.set_alpha(0)
        ax2.spines["right"].set_visible(False)

    ax.set_title(col, fontsize=11, pad=10)
    ax.set_xlabel("Value", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

for j in range(n_features_test, len(axes)):
    axes[j].axis("off")

fig.suptitle("Numerical feature distributions (test set)", fontsize=16)
fig.tight_layout(rect=[0, 0.0, 1, 0.96])
fig.savefig(OUT_FIG_NUM_TEST, format="svg", dpi=200)
plt.close(fig)

######### Time series: order date (DateOrders) #########
if DATE_COL in train_df.columns:
    OUT_FIG_DATE = OUTPUT_DIR / "02_order_date_timeseries.svg"

    train_dates = pd.to_datetime(train_df[DATE_COL], errors="coerce").dropna()
    test_dates = pd.to_datetime(test_df[DATE_COL], errors="coerce").dropna()

    train_counts = train_dates.dt.to_period("D").value_counts().sort_index()
    test_counts = test_dates.dt.to_period("D").value_counts().sort_index()

    train_counts.index = train_counts.index.to_timestamp()
    test_counts.index = test_counts.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_counts.index, train_counts.values, color="#4472C4", linewidth=1.5, label="Train")
    ax.plot(test_counts.index, test_counts.values, color="#FF7F0E", linewidth=1.5, label="Test")

    ax.set_title("Orders over time (daily counts)", fontsize=13, pad=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_FIG_DATE, format="svg", dpi=200)
    plt.close(fig)

######### Categorical feature distributions (train only) #########
# We keep the visualization focused on the training set, which is what
# we use to fit models and transformations.

categorical_cols = (
    train_df.select_dtypes(include=["object", "category", "bool", "string"])
    .columns.tolist()
)

# Plot `order date (DateOrders)` separately as time series
categorical_cols = [c for c in categorical_cols if c != DATE_COL]

# Treat "Order Item Quantity" as categorical for EDA
if "Order Item Quantity" in train_df.columns and "Order Item Quantity" not in categorical_cols:
    categorical_cols.append("Order Item Quantity")

if not categorical_cols:
    raise ValueError("No categorical columns found in train_df for EDA.")

OUT_FIG_CAT_TRAIN = OUTPUT_DIR / "01_categorical_distribution.svg"

n_cat = len(categorical_cols)
ncols_cat = 3
nrows_cat = math.ceil(n_cat / ncols_cat)

fig, axes = plt.subplots(
    nrows=nrows_cat,
    ncols=ncols_cat,
    figsize=(6.6 * ncols_cat, 4.0 * nrows_cat),
)
axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, col in enumerate(categorical_cols):
    ax = axes[i]

    series = train_df[col].fillna("<NA>")
    value_counts = series.value_counts()
    n_unique = series.nunique(dropna=False)

    if len(value_counts) > 6:
        top = value_counts.head(6)
        other = value_counts.iloc[6:].sum()
        if other > 0:
            top.loc["Other"] = other
        counts = top
    else:
        counts = value_counts

    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    bar_colors = sns.color_palette("Blues", n_colors=len(counts))
    ax.bar(
        x=range(len(counts)),
        height=counts.values,
        color=bar_colors,
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.astype(str), rotation=30, ha="right")

    for idx, (label, y_val, pct) in enumerate(
        zip(counts.index.astype(str), counts.values, percentages.values)
    ):
        ax.text(
            idx,
            y_val,
            f"{pct} %",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(col, fontsize=11, pad=10)
    if len(value_counts) > 6:
        ax.text(
            0.50,
            0.98,
            f"Unique: {n_unique}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
        )
    ax.set_xlabel("Category", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")

for j in range(n_cat, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "Categorical feature distributions (train set)",
    fontsize=16,
)
fig.subplots_adjust(hspace=0.6, wspace=0.35)
fig.tight_layout(rect=[0, 0.0, 1, 0.96])
fig.savefig(OUT_FIG_CAT_TRAIN, format="svg", dpi=200)
plt.close(fig)

######### Categorical feature distributions (test) #########
OUT_FIG_CAT_TEST = OUTPUT_DIR / "01_categorical_distribution_test.svg"

fig, axes = plt.subplots(
    nrows=nrows_cat,
    ncols=ncols_cat,
    figsize=(6.6 * ncols_cat, 4.0 * nrows_cat),
)
axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, col in enumerate(categorical_cols):
    ax = axes[i]

    series = test_df[col].fillna("<NA>")
    value_counts = series.value_counts()
    n_unique = series.nunique(dropna=False)

    if len(value_counts) > 6:
        top = value_counts.head(6)
        other = value_counts.iloc[6:].sum()
        if other > 0:
            top.loc["Other"] = other
        counts = top
    else:
        counts = value_counts

    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    bar_colors = sns.color_palette("Blues", n_colors=len(counts))
    ax.bar(
        x=range(len(counts)),
        height=counts.values,
        color=bar_colors,
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.astype(str), rotation=30, ha="right")

    for idx, (label, y_val, pct) in enumerate(
        zip(counts.index.astype(str), counts.values, percentages.values)
    ):
        ax.text(
            idx,
            y_val,
            f"{pct} %",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(col, fontsize=11, pad=10)
    if len(value_counts) > 6:
        ax.text(
            0.50,
            0.98,
            f"Unique: {n_unique}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
        )
    ax.set_xlabel("Category", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")

for j in range(n_cat, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "Categorical feature distributions (test set)",
    fontsize=16,
)
fig.subplots_adjust(hspace=0.6, wspace=0.35)
fig.tight_layout(rect=[0, 0.0, 1, 0.96])
fig.savefig(OUT_FIG_CAT_TEST, format="svg", dpi=200)
plt.close(fig)

######### ydata-profiling EDA report (train set) #########
# We build a copy of train_df and ensure categorical information is explicit
# before generating the profile.

eda_df = train_df.copy()

for col in eda_df.columns:
    col_series = eda_df[col]
    if col_series.dtype.name in ("object", "category", "bool", "string"):
        eda_df[col] = col_series.astype("category")
    else:
        n_unique = col_series.nunique(dropna=False)
        if n_unique <= 15:
            eda_df[col] = col_series.astype("category")

profile = ProfileReport(
    eda_df,
    title="SMARTCO Delivery Status - Train EDA",
    explorative=True,
)
profile_output_path = OUTPUT_DIR / "03_ydata_profile_train.html"
profile.to_file(str(profile_output_path))

