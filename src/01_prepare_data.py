"""
This script performs Data Leakage analysis and executes the train-test split 
to ensure a clean separation for model evaluation.
"""
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

TEST_SIZE = 0.2
DATA_PATH = Path(__file__).parent.parent / "data"

######### Load the dataset #########
df = pd.read_csv(os.path.join(DATA_PATH, "raw", "DataCoSupplyChainDataset.csv"), encoding="latin-1")

######### Study Number of NAs #########
na_summary = (
    df.isna()
      .sum()
      .loc[lambda x: x > 1]
      .to_frame("na_count")
      .assign(na_percent=lambda x: 100 * x["na_count"] / len(df))
      .sort_values("na_percent", ascending=False)
)
print("NA Summary:")
print(na_summary)
print()

# We are going to drop the "Product Description" and "Order Zipcode" columns as they have a 100% and 86% NA values respectively.
df = df.drop(columns=["Product Description", "Order Zipcode"])

######### Duplicates #########
n_duplicates = df.duplicated().sum()
percent = 100 * n_duplicates / len(df)

print(f"Duplicated rows: {n_duplicates} ({percent:.2f}%)")
print()
# There aren't any duplicated rows in the dataset.

######### Data Leakage and Noise Columns #########
LEAKAGE_COLS = [
    "Days for shipping (real)", 
    "Late_delivery_risk", 
    "shipping date (DateOrders)",
    "Order Status"
]

PII_AND_IDS_COLS = [
    "Customer Password", "Customer Email", 
    "Customer Fname", "Customer Lname",
    "Customer Id", "Order Id",
    "Order Item Id", "Order Customer Id",
    "Product Card Id", "Order Item Cardprod Id"
]

TEXT_AND_NOISE_COLS = [
    "Product Image", 
    "Customer Street"
]

REDUNDANT_COLS = [
    "Category Id",             # We keep Category Name
    "Department Id",           # We keep Department Name
    "Product Category Id",
    "Order Profit Per Order",  # same as Benefit per order
    "Order Item Product Price",  # same as Product Price
    "Order Item Total",        # same as Sales per customer
    "Customer Zipcode",        # we already keep city and state
    "Product Status",          # constant, non-informative
]

# Unimos todas las listas para el drop final
ALL_COLS_TO_DROP = LEAKAGE_COLS + PII_AND_IDS_COLS + TEXT_AND_NOISE_COLS + REDUNDANT_COLS

# Aplicamos la limpieza
df = df.drop(columns=ALL_COLS_TO_DROP)

print("DF shape after dropping columns:", df.shape)

######### Train Test Split #########
# As we have a time component in the dataset, we will perform a time-based split to avoid data leakage.
df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])
df = df.sort_values("order date (DateOrders)")
split_index = int(len(df) * (1 - TEST_SIZE))
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]
print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Saving the cleaned datasets
os.makedirs(os.path.join(DATA_PATH, "processed"), exist_ok=True)
train_df.to_csv(os.path.join(DATA_PATH, "processed", "train_raw.csv"), index=False)
test_df.to_csv(os.path.join(DATA_PATH, "processed", "test_raw.csv"), index=False)
