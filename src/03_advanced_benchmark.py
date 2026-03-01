"""
This script runs an advanced benchmark for Delivery Status prediction
using multiple models and configurable class-imbalance strategies.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Delivery Status"
DATE_COL = "order date (DateOrders)"
DATA_PATH = Path(__file__).parent.parent / "data"
RANDOM_STATE = 42

# Options: "none", "class_weight", "undersample", "oversample", "hybrid"
IMBALANCE_STRATEGY = "hybrid"


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple calendar features from order date."""
    df = df.copy()

    if DATE_COL not in df.columns:
        return df

    dt = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["order_year"] = dt.dt.year
    df["order_month"] = dt.dt.month
    df["order_day"] = dt.dt.day
    df["order_weekday"] = dt.dt.weekday
    df["order_is_weekend"] = (dt.dt.weekday >= 5).astype(float)

    # Cyclical encoding for month and weekday
    df["order_month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12)
    df["order_month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12)
    df["order_weekday_sin"] = np.sin(2 * np.pi * df["order_weekday"] / 7)
    df["order_weekday_cos"] = np.cos(2 * np.pi * df["order_weekday"] / 7)

    df = df.drop(columns=[DATE_COL])
    return df


def apply_sampling_strategy(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply under/over/hybrid sampling in the training set."""
    if strategy not in {"undersample", "oversample", "hybrid"}:
        return X, y

    data = X.copy()
    data[TARGET_COL] = y.values
    class_counts = y.value_counts()

    if strategy == "undersample":
        target_n = int(class_counts.min())
    elif strategy == "oversample":
        target_n = int(class_counts.max())
    else:
        target_n = int(class_counts.median())

    sampled_parts = []
    for class_name, class_df in data.groupby(TARGET_COL):
        current_n = len(class_df)

        if current_n > target_n:
            sampled_df = class_df.sample(n=target_n, replace=False, random_state=random_state)
        elif current_n < target_n:
            sampled_df = class_df.sample(n=target_n, replace=True, random_state=random_state)
        else:
            sampled_df = class_df

        sampled_parts.append(sampled_df)

    balanced_data = pd.concat(sampled_parts, axis=0)
    balanced_data = balanced_data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    X_balanced = balanced_data.drop(columns=[TARGET_COL])
    y_balanced = balanced_data[TARGET_COL]
    return X_balanced, y_balanced


def build_model_zoo(class_weight_map: dict | None) -> dict[str, object]:
    """Create model candidates for advanced benchmark."""
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            verbose=1,
            max_iter=4000,
            solver="saga",
            class_weight=class_weight_map,
            tol=1e-2,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=class_weight_map,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=class_weight_map,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            objective="multi:softprob",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception:
        print("xgboost is not available. Skipping xgboost model.")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            objective="multiclass",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=class_weight_map,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        print("lightgbm is not available. Skipping lightgbm model.")

    return models


######### Load train/test datasets #########
train_df = pd.read_csv(os.path.join(DATA_PATH, "processed", "train_raw.csv"))
test_df = pd.read_csv(os.path.join(DATA_PATH, "processed", "test_raw.csv"))

######### Feature engineering #########
train_df = add_date_features(train_df)
test_df = add_date_features(test_df)

######### Define features and target #########
if TARGET_COL not in train_df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' was not found in train dataset.")
if TARGET_COL not in test_df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' was not found in test dataset.")

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

######### Class imbalance strategy #########
print(f"Using imbalance strategy: {IMBALANCE_STRATEGY}")

X_train_used, y_train_used = apply_sampling_strategy(
    X_train,
    y_train,
    strategy=IMBALANCE_STRATEGY,
    random_state=RANDOM_STATE,
)

class_weight_map = None
if IMBALANCE_STRATEGY == "class_weight":
    class_weight_map = "balanced"

######### Preprocessing #########
numeric_cols = X_train_used.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = X_train_used.select_dtypes(exclude=["number", "bool"]).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=20,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
)

######### Train and evaluate model zoo #########
models = build_model_zoo(class_weight_map=class_weight_map)
results = []

for model_name, model in models.items():
    print("\n" + "=" * 60)
    print(f"Training model: {model_name}")

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    try:
        clf.fit(X_train_used, y_train_used)
    except Exception as exc:
        print(f"Model '{model_name}' failed during training: {exc}")
        continue

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(
        DATA_PATH,
        "processed",
        f"advanced_{model_name}_classification_report.csv",
    )
    report_df.to_csv(report_path, index=True)

    preds_df = test_df[[TARGET_COL]].copy()
    preds_df["prediction"] = y_pred
    preds_path = os.path.join(
        DATA_PATH,
        "processed",
        f"advanced_{model_name}_predictions.csv",
    )
    preds_df.to_csv(preds_path, index=False)

    results.append(
        {
            "model": model_name,
            "imbalance_strategy": IMBALANCE_STRATEGY,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "classification_report_path": report_path,
            "predictions_path": preds_path,
        }
    )

if not results:
    raise RuntimeError("All advanced benchmark models failed. Please review dependency/setup logs.")

######### Save leaderboard #########
results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
leaderboard_path = os.path.join(DATA_PATH, "processed", "advanced_benchmark_leaderboard.csv")
results_df.to_csv(leaderboard_path, index=False)

print("\n" + "=" * 60)
print("Advanced benchmark completed.")
print(f"Leaderboard saved to: {leaderboard_path}")
print(results_df[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]])
