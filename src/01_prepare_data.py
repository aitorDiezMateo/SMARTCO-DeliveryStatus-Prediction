"""
This script performs Data Leakage analysis and executes the train-test split 
to ensure a clean separation for model evaluation.
"""
import pandas as pd
import os
from pathlib import Path
import numpy as np

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

######### Feature Engineering: Temporal Target Encoding #########
print("Generating rolling target encoding features...")
# Asegurarse de que son datetime para calcular fechas
df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'])

# Fecha de llegada (real)
df['Arrival Date Real'] = df['shipping date (DateOrders)'] + pd.to_timedelta(df['Days for shipping (real)'], unit='D')

# Extraer el historial de llegadas y hacer dummies del target
df_arrivals = df[['Arrival Date Real', 'Delivery Status']].dropna().sort_values('Arrival Date Real')
dummies = pd.get_dummies(df_arrivals['Delivery Status'])
df_arrivals = pd.concat([df_arrivals, dummies], axis=1)

# Agrupar llegadas por dia
daily_arrivals = df_arrivals.groupby(df_arrivals['Arrival Date Real'].dt.floor('D')).sum(numeric_only=True)

# Reindexar para cubrir todo el lapso de tiempo hasta la fecha maxima
full_date_range = pd.date_range(daily_arrivals.index.min(), df['order date (DateOrders)'].max().floor('D'))
daily_arrivals = daily_arrivals.reindex(full_date_range).fillna(0)

# Calcular porcentajes rolling sin data leakage
windows = [7, 14, 30]
rolling_features = {}

for w in windows:
    # shift(1) es super importante: asegura que los conteos sean estrictamente anteriores al dia actual (evita fuga de datos del mismo dia futuro)
    roll_sum = daily_arrivals.rolling(window=w, min_periods=1).sum().shift(1)
    total = roll_sum.sum(axis=1).replace(0, np.nan)
    pct = roll_sum.div(total, axis=0).fillna(0)
    
    for col in pct.columns:
        col_name = str(col).replace(" ", "_").lower()
        rolling_features[f'target_enc_{col_name}_{w}d'] = pct[col]

rolling_df = pd.DataFrame(rolling_features)

rolling_df = rolling_df.fillna(0)  # In case there are any NaNs left (e.g., at the very beginning of the series)

# Unir con el dataset principal truncando las fechas a nivel de dia
df['order_date_floor'] = df['order date (DateOrders)'].dt.floor('D')
df = df.merge(rolling_df, left_on='order_date_floor', right_index=True, how='left')

# Rellenar NaNs solo en las columnas de target encoding tras el merge
target_enc_cols = [col for col in df.columns if col.startswith('target_enc_')]
df[target_enc_cols] = df[target_enc_cols].fillna(0)

# Drop columnas temporales
df = df.drop(columns=['Arrival Date Real', 'order_date_floor'])
print("Rolling target encoding features generated successfully.")
print("Current shape with new features:", df.shape)
print()

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
# order date has already been converted to datetime during Feature Engineering
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
