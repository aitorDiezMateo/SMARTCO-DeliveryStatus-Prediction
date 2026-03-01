"""
This script trains a benchmark Logistic Regression model to predict
the Delivery Status target from the processed train/test datasets.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Delivery Status"
DATE_COL = "order date (DateOrders)"
DATA_PATH = Path(__file__).parent.parent / "data"
RANDOM_STATE = 42


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

######### Preprocessing #########
numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()

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

######### Model #########
model = LogisticRegression(
    verbose=1,
    max_iter=10000,
    solver="saga",
    class_weight="balanced",
    tol=1e-2,
    random_state=RANDOM_STATE,
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

######### Training #########
clf.fit(X_train, y_train)

######### Evaluation #########
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)

f1 = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["f1-score"]

print("Benchmark Model: Logistic Regression")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

######### Save classification report #########
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(DATA_PATH, "processed", "benchmark_classification_report.csv"), index=True)
print("\nClassification report saved to data/processed/benchmark_classification_report.csv")

######### Save predictions #########
preds_df = test_df[[TARGET_COL]].copy()
preds_df["prediction"] = y_pred
preds_df.to_csv(os.path.join(DATA_PATH, "processed", "benchmark_predictions.csv"), index=False)
print("\nPredictions saved to data/processed/benchmark_predictions.csv")
