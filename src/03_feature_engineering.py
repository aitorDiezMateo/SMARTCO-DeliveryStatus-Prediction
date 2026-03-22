"""
EXPLORATORY FEATURE ENGINEERING ANALYSIS

This script is NOT integrated into the final prediction pipeline. Instead, it serves
as a documented exploration and visualization of feature engineering techniques,
including:
- Power transformations (Yeo-Johnson) with before/after KDE comparisons
- Feature discretization/binning strategies
- Handcrafted feature engineering decisions

The ACTUAL production feature pipeline is in src/pipelines/feature_engineering.py,
which is used by all training scripts (XGBoost, CatBoost, Bagging, Stacking, Voting).

Use this script to understand the methodology and generate visualizations for reports.
Outputs are saved to output/feature_engineering/ (SVGs and diagnostic CSVs).
"""

import os
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
import numpy as np

DATA_PATH = Path(__file__).parent.parent / "data"
TRAIN_RAW_PATH = os.path.join(DATA_PATH, "processed", "train_raw.csv")
TEST_RAW_PATH = os.path.join(DATA_PATH, "processed", "test_raw.csv")

######### Load processed datasets #########
train_df = pd.read_csv(TRAIN_RAW_PATH)
test_df = pd.read_csv(TEST_RAW_PATH)

######### Numerical features configuration #########
NUMERICAL_FEATURES = [
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Profit Ratio",
    "Sales",
    "Product Price",
]

missing_cols = [c for c in NUMERICAL_FEATURES if c not in train_df.columns]
if missing_cols:
    raise ValueError(
        "Missing expected numerical columns in train_df: "
        + ", ".join(missing_cols)
        + f". Available columns: {len(train_df.columns)}"
    )

######### Numerical dataset (train only) #########
numerical_train_df = train_df[NUMERICAL_FEATURES].copy()
for c in NUMERICAL_FEATURES:
    numerical_train_df[c] = pd.to_numeric(numerical_train_df[c], errors="coerce")

######### Numerical dataset (test only) #########
numerical_test_df = test_df[NUMERICAL_FEATURES].copy()
for c in NUMERICAL_FEATURES:
    numerical_test_df[c] = pd.to_numeric(numerical_test_df[c], errors="coerce")

######### Feature engineering output directory #########
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "feature_engineering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

######### Power transformations (Yeo-Johnson) #########
POWER_TRANSFORM_FEATURES = [
    "Benefit per order",
    "Order Item Discount",
    "Sales per customer",
    "Order Item Profit Ratio",
]

missing_power_cols = [c for c in POWER_TRANSFORM_FEATURES if c not in numerical_train_df.columns]
if missing_power_cols:
    raise ValueError(
        "Missing expected columns for power transformation: "
        + ", ".join(missing_power_cols)
    )

# Use Yeo-Johnson without standardization so the transformation is purely
# monotonic; this makes the shape comparison more interpretable.
pt = PowerTransformer(method="yeo-johnson", standardize=False)
power_data = numerical_train_df[POWER_TRANSFORM_FEATURES].copy()
transformed_values = pt.fit_transform(power_data)
transformed_df = pd.DataFrame(
    transformed_values,
    columns=POWER_TRANSFORM_FEATURES,
    index=power_data.index,
)

test_power_data = numerical_test_df[POWER_TRANSFORM_FEATURES].copy()
test_transformed_values = pt.transform(test_power_data)
test_transformed_df = pd.DataFrame(
    test_transformed_values,
    columns=POWER_TRANSFORM_FEATURES,
    index=test_power_data.index,
)

######### KDE comparison: original vs Yeo-Johnson transformed (train) #########
OUT_FIG_POWER_PATH = OUTPUT_DIR / "01_power_transformations_kde.svg"

sns.set_theme(style="whitegrid")

n_features_power = len(POWER_TRANSFORM_FEATURES)

fig, axes = plt.subplots(
    nrows=n_features_power,
    ncols=2,
    figsize=(10.0, 2.6 * n_features_power),
    constrained_layout=True,
)

for i, col in enumerate(POWER_TRANSFORM_FEATURES):
    ax_orig = axes[i, 0]
    ax_trans = axes[i, 1]

    x_orig = power_data[col].dropna()
    x_trans = transformed_df[col].dropna()

    # Original distribution
    sns.kdeplot(
        x=x_orig,
        ax=ax_orig,
        fill=True,
        color="#4472C4",
        alpha=0.35,
        linewidth=1.5,
    )
    ax_orig.set_title(f"Original - {col}", fontsize=11, pad=10)
    ax_orig.set_xlabel("Value", fontsize=9)
    ax_orig.set_ylabel("Density", fontsize=9)
    ax_orig.tick_params(axis="both", labelsize=8)
    ax_orig.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    # Transformed distribution
    sns.kdeplot(
        x=x_trans,
        ax=ax_trans,
        fill=True,
        color="#FF7F0E",
        alpha=0.35,
        linewidth=1.5,
    )
    ax_trans.set_title(f"Yeo-Johnson - {col}", fontsize=11, pad=10)
    ax_trans.set_xlabel("Transformed value", fontsize=9)
    ax_trans.set_ylabel("Density", fontsize=9)
    ax_trans.tick_params(axis="both", labelsize=8)
    ax_trans.grid(alpha=0.25, linestyle="--", linewidth=0.5)

fig.suptitle(
    "Original vs Yeo-Johnson transformed distributions (train set)",
    fontsize=16,
    y=1.02,
)
fig.savefig(OUT_FIG_POWER_PATH, format="svg", dpi=200)
plt.close(fig)

######### KDE comparison: original vs Yeo-Johnson transformed (test) #########
OUT_FIG_POWER_PATH_TEST = OUTPUT_DIR / "01_power_transformations_kde_test.svg"

fig, axes = plt.subplots(
    nrows=n_features_power,
    ncols=2,
    figsize=(10.0, 2.6 * n_features_power),
    constrained_layout=True,
)

for i, col in enumerate(POWER_TRANSFORM_FEATURES):
    ax_orig = axes[i, 0]
    ax_trans = axes[i, 1]

    x_orig = test_power_data[col].dropna()
    x_trans = test_transformed_df[col].dropna()

    sns.kdeplot(
        x=x_orig,
        ax=ax_orig,
        fill=True,
        color="#4472C4",
        alpha=0.35,
        linewidth=1.5,
    )
    ax_orig.set_title(f"Original (test) - {col}", fontsize=11, pad=10)
    ax_orig.set_xlabel("Value", fontsize=9)
    ax_orig.set_ylabel("Density", fontsize=9)
    ax_orig.tick_params(axis="both", labelsize=8)
    ax_orig.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    sns.kdeplot(
        x=x_trans,
        ax=ax_trans,
        fill=True,
        color="#FF7F0E",
        alpha=0.35,
        linewidth=1.5,
    )
    ax_trans.set_title(f"Yeo-Johnson (test) - {col}", fontsize=11, pad=10)
    ax_trans.set_xlabel("Transformed value", fontsize=9)
    ax_trans.set_ylabel("Density", fontsize=9)
    ax_trans.tick_params(axis="both", labelsize=8)
    ax_trans.grid(alpha=0.25, linestyle="--", linewidth=0.5)

fig.suptitle(
    "Original vs Yeo-Johnson transformed distributions (test set)",
    fontsize=16,
    y=1.02,
)
fig.savefig(OUT_FIG_POWER_PATH_TEST, format="svg", dpi=200)
plt.close(fig)

######### Discretization (binning) for prices and sales #########
DISCRETIZATION_FEATURES_PRICE = [
    "Product Price",
]
DISCRETIZATION_FEATURE_SALES = "Sales"

price_bins = [0, 50, 100, 200, 300, 400, math.inf]
price_bin_labels = [
    "<=50",
    "50-100",
    "100-200",
    "200-300",
    "300-400",
    ">=400",
]

# Price-based bins
for col in DISCRETIZATION_FEATURES_PRICE:
    if col not in numerical_train_df.columns:
        raise ValueError(f"Missing expected column for price discretization: {col}")
    series = numerical_train_df[col]
    binned = pd.cut(
        series,
        bins=price_bins,
        labels=price_bin_labels,
        include_lowest=True,
        right=False,
    )
    numerical_train_df[f"{col} Bin"] = binned

    if col not in numerical_test_df.columns:
        raise ValueError(f"Missing expected column for price discretization (test): {col}")
    test_series = numerical_test_df[col]
    test_binned = pd.cut(
        test_series,
        bins=price_bins,
        labels=price_bin_labels,
        include_lowest=True,
        right=False,
    )
    numerical_test_df[f"{col} Bin"] = test_binned

# Sales bins using quantiles
if DISCRETIZATION_FEATURE_SALES not in numerical_train_df.columns:
    raise ValueError(
        f"Missing expected column for sales discretization: {DISCRETIZATION_FEATURE_SALES}"
    )

sales_series = numerical_train_df[DISCRETIZATION_FEATURE_SALES]
sales_bins = pd.qcut(
    sales_series,
    q=5,
    duplicates="drop",
)

intervals = sales_bins.cat.categories
sales_bin_edges = [intervals[0].left] + [iv.right for iv in intervals]

sales_categories = [f"Q{i+1}" for i in range(len(intervals))]
sales_bins = sales_bins.cat.rename_categories(sales_categories)
numerical_train_df["Sales Bin"] = sales_bins

test_sales_series = numerical_test_df[DISCRETIZATION_FEATURE_SALES]
test_sales_bins = pd.cut(
    test_sales_series,
    bins=sales_bin_edges,
    labels=sales_categories,
    include_lowest=True,
)
numerical_test_df["Sales Bin"] = test_sales_bins

######### Visualization of discretization effect #########
OUT_FIG_DISCR_PATH = OUTPUT_DIR / "02_discretization_price_sales.svg"
OUT_FIG_DISCR_PATH_TEST = OUTPUT_DIR / "02_discretization_price_sales_test.svg"

sns.set_theme(style="whitegrid")

disc_pairs = [
    ("Product Price", "Product Price Bin"),
    ("Sales", "Sales Bin"),
]

n_disc = len(disc_pairs)

fig, axes = plt.subplots(
    nrows=n_disc,
    ncols=2,
    figsize=(11.0, 2.8 * n_disc),
)

for i, (cont_col, bin_col) in enumerate(disc_pairs):
    ax_left = axes[i, 0]
    ax_right = axes[i, 1]

    x_cont = numerical_train_df[cont_col].dropna()

    # Left: original continuous distribution
    ax_left.hist(
        x_cont,
        bins=40,
        color="#4472C4",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )
    ax_left.set_title(f"Original - {cont_col}", fontsize=11, pad=10)
    ax_left.set_xlabel("Value", fontsize=9)
    ax_left.set_ylabel("Count", fontsize=9)
    ax_left.tick_params(axis="both", labelsize=8)
    ax_left.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    # Right: discretized counts per bin (drop empty bins so we don't plot 0.0 % categories)
    bin_series = numerical_train_df[bin_col].dropna()
    counts = bin_series.value_counts().sort_index()
    counts = counts[counts > 0]
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    # Use Matplotlib bar to avoid seaborn palette/hue deprecation warnings
    bar_colors = sns.color_palette("Blues", n_colors=len(counts))
    ax_right.bar(
        x=range(len(counts)),
        height=counts.values,
        color=bar_colors,
    )
    ax_right.set_xticks(range(len(counts)))
    ax_right.set_xticklabels(counts.index.astype(str))
    for idx, (x_label, y_val, pct) in enumerate(
        zip(counts.index.astype(str), counts.values, percentages.values)
    ):
        ax_right.text(
            idx,
            y_val,
            f"{pct} %",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_right.set_title(f"Discretized - {cont_col}", fontsize=11, pad=10)
    ax_right.set_xlabel("Bin", fontsize=9)
    ax_right.set_ylabel("Count", fontsize=9)
    ax_right.tick_params(axis="both", labelsize=8)
    ax_right.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")

fig.suptitle(
    "Price and sales discretization (original vs binned distributions)",
    fontsize=16,
)
fig.tight_layout(rect=[0, 0.0, 1, 0.99])
fig.savefig(OUT_FIG_DISCR_PATH, format="svg", dpi=300)
plt.close(fig)

######### Visualization of discretization effect (test) #########
fig, axes = plt.subplots(
    nrows=n_disc,
    ncols=2,
    figsize=(11.0, 2.8 * n_disc),
)

for i, (cont_col, bin_col) in enumerate(disc_pairs):
    ax_left = axes[i, 0]
    ax_right = axes[i, 1]

    x_cont = numerical_test_df[cont_col].dropna()

    ax_left.hist(
        x_cont,
        bins=40,
        color="#4472C4",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )
    ax_left.set_title(f"Original (test) - {cont_col}", fontsize=11, pad=10)
    ax_left.set_xlabel("Value", fontsize=9)
    ax_left.set_ylabel("Count", fontsize=9)
    ax_left.tick_params(axis="both", labelsize=8)
    ax_left.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    bin_series = numerical_test_df[bin_col].dropna()
    counts = bin_series.value_counts().sort_index()
    counts = counts[counts > 0]
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    bar_colors = sns.color_palette("Blues", n_colors=len(counts))
    ax_right.bar(
        x=range(len(counts)),
        height=counts.values,
        color=bar_colors,
    )
    ax_right.set_xticks(range(len(counts)))
    ax_right.set_xticklabels(counts.index.astype(str))

    for idx, (x_label, y_val, pct) in enumerate(
        zip(counts.index.astype(str), counts.values, percentages.values)
    ):
        ax_right.text(
            idx,
            y_val,
            f"{pct} %",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_right.set_title(f"Discretized (test) - {cont_col}", fontsize=11, pad=10)
    ax_right.set_xlabel("Bin", fontsize=9)
    ax_right.set_ylabel("Count", fontsize=9)
    ax_right.tick_params(axis="both", labelsize=8)
    ax_right.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")

fig.suptitle(
    "Price and sales discretization (test set, original vs binned distributions)",
    fontsize=16,
)
fig.tight_layout(rect=[0, 0.0, 1, 0.99])
fig.savefig(OUT_FIG_DISCR_PATH_TEST, format="svg", dpi=300)
plt.close(fig)

######### New handcrafted feature engineering #########
DATE_COL = "order date (DateOrders)"

if DATE_COL not in train_df.columns:
    raise ValueError(f"Expected date column '{DATE_COL}' not found in train_df.")

######### Date-based features #########
train_dates = pd.to_datetime(train_df[DATE_COL], errors="coerce")
test_dates = pd.to_datetime(test_df[DATE_COL], errors="coerce")

for df, dates, prefix in [
    (train_df, train_dates, "train"),
    (test_df, test_dates, "test"),
]:
    df["order_year"] = dates.dt.year
    df["order_month"] = dates.dt.month
    df["order_day"] = dates.dt.day
    df["is_weekend"] = dates.dt.weekday.isin([5, 6]).astype(int)

######### Cross-border shipping feature #########
for df in [train_df, test_df]:
    if "Customer Country" not in df.columns or "Order Country" not in df.columns:
        raise ValueError("Expected 'Customer Country' and 'Order Country' columns.")
    df["is_cross_border"] = (df["Customer Country"] != df["Order Country"]).astype(int)

######### Shipping route feature #########
for df in [train_df, test_df]:
    if "Order Region" not in df.columns or "Market" not in df.columns:
        raise ValueError("Expected 'Order Region' and 'Market' columns.")
    origin = df["Order Region"].astype(str).str.replace(" ", "_")
    destination = df["Market"].astype(str).str.replace(" ", "_")
    df["shipping_route"] = origin + "_to_" + destination

######### One-hot encoding for low-cardinality categorical features #########
LOW_CARD_COLS = ["Type", "Shipping Mode", "Customer Segment", "Department Name"]

for col in LOW_CARD_COLS:
    if col not in train_df.columns:
        raise ValueError(f"Expected low-cardinality feature '{col}' not found in train_df.")

train_low = train_df[LOW_CARD_COLS].fillna("<NA>")
test_low = test_df[LOW_CARD_COLS].fillna("<NA>")

train_low_dummies = pd.get_dummies(train_low, prefix=LOW_CARD_COLS)
test_low_dummies = pd.get_dummies(test_low, prefix=LOW_CARD_COLS)

test_low_dummies = test_low_dummies.reindex(columns=train_low_dummies.columns, fill_value=0)

train_df = pd.concat([train_df, train_low_dummies], axis=1)
test_df = pd.concat([test_df, test_low_dummies], axis=1)

######### Target encoding for high-cardinality features #########
HIGH_CARD_COLS = ["Order City", "Customer City", "Product Name", "Category Name"]

for col in HIGH_CARD_COLS:
    if col not in train_df.columns:
        raise ValueError(f"Expected high-cardinality feature '{col}' not found in train_df.")

if "Delivery Status" not in train_df.columns:
    raise ValueError("Expected target column 'Delivery Status' not found in train_df.")

target_binary = (train_df["Delivery Status"] == "Late delivery").astype(float)
global_mean = target_binary.mean()

for col in HIGH_CARD_COLS:
    means = target_binary.groupby(train_df[col]).mean()
    enc_col = f"{col}_target_enc"
    train_df[enc_col] = train_df[col].map(means).fillna(global_mean)
    test_df[enc_col] = test_df[col].map(means).fillna(global_mean)

######### Plot engineered numeric features (train & test) #########
ENGINEERED_NUM_FEATURES = [
    "order_year",
    "order_month",
    "order_day",
    "is_weekend",
    "Order City_target_enc",
    "Customer City_target_enc",
    "Product Name_target_enc",
    "Category Name_target_enc",
]

# Include is_cross_border only if it is not a constant 1.0 feature in train
if "is_cross_border" in train_df.columns:
    unique_cross = train_df["is_cross_border"].dropna().unique()
    if not (len(unique_cross) == 1 and unique_cross[0] == 1.0):
        ENGINEERED_NUM_FEATURES.insert(4, "is_cross_border")

missing_eng_train = [c for c in ENGINEERED_NUM_FEATURES if c not in train_df.columns]
if missing_eng_train:
    raise ValueError(
        "Missing engineered numeric features in train_df: " + ", ".join(missing_eng_train)
    )

missing_eng_test = [c for c in ENGINEERED_NUM_FEATURES if c not in test_df.columns]
if missing_eng_test:
    raise ValueError(
        "Missing engineered numeric features in test_df: " + ", ".join(missing_eng_test)
    )

sns.set_theme(style="whitegrid")

OUT_FIG_FE_NUM_TRAIN = OUTPUT_DIR / "03_engineered_numeric_features_train.svg"
OUT_FIG_FE_NUM_TEST = OUTPUT_DIR / "03_engineered_numeric_features_test.svg"

def _plot_engineered_numeric(df: pd.DataFrame, out_path: Path, title: str) -> None:
    n_features = len(ENGINEERED_NUM_FEATURES)
    ncols = 3
    nrows = math.ceil(n_features / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 3.5 * nrows),
        constrained_layout=True,
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    palette_color = "#4C78A8"

    for i, col in enumerate(ENGINEERED_NUM_FEATURES):
        ax = axes[i]
        x = pd.to_numeric(df[col], errors="coerce").dropna()

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

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.savefig(out_path, format="svg", dpi=200)
    plt.close(fig)


_plot_engineered_numeric(train_df, OUT_FIG_FE_NUM_TRAIN, "Engineered numeric features (train set)")
_plot_engineered_numeric(test_df, OUT_FIG_FE_NUM_TEST, "Engineered numeric features (test set)")

######### Save feature-engineered datasets #########
FE_DATA_PATH = DATA_PATH / "feature_engineering"
FE_DATA_PATH.mkdir(parents=True, exist_ok=True)

train_fe_path = FE_DATA_PATH / "train_features.csv"
test_fe_path = FE_DATA_PATH / "test_features.csv"

train_df.to_csv(train_fe_path, index=False)
test_df.to_csv(test_fe_path, index=False)

