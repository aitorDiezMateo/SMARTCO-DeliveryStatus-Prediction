"""
General, CV-safe feature engineering + preprocessing pipeline.

- Deterministic feature engineering is always the same (FeatureBuilder).
- Encoders / scalers / oversamplers / models are configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

DATE_COL_DEFAULT = "order date (DateOrders)"
TARGET_COL_DEFAULT = "Delivery Status"


def _safe_series(x: pd.Series) -> pd.Series:
    if pd.api.types.is_categorical_dtype(x):
        return x.astype(str)
    return x


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Adds deterministic features (date parts, weekend, cross-border, route).
    """

    def __init__(
        self,
        date_col: str = DATE_COL_DEFAULT,
        customer_country_col: str = "Customer Country",
        order_country_col: str = "Order Country",
        order_region_col: str = "Order Region",
        market_col: str = "Market",
    ) -> None:
        self.date_col = date_col
        self.customer_country_col = customer_country_col
        self.order_country_col = order_country_col
        self.order_region_col = order_region_col
        self.market_col = market_col
        self._include_cross_border: bool = True

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ANN001
        X = X.copy()
        if self.customer_country_col not in X.columns or self.order_country_col not in X.columns:
            raise ValueError(
                f"Expected '{self.customer_country_col}' and '{self.order_country_col}' columns."
            )
        cross = (X[self.customer_country_col] != X[self.order_country_col]).astype(float)
        uniq = pd.unique(cross.dropna())
        self._include_cross_border = not (len(uniq) == 1 and float(uniq[0]) == 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.date_col not in X.columns:
            raise ValueError(f"Expected date column '{self.date_col}' not found.")

        d = pd.to_datetime(X[self.date_col], errors="coerce")
        X["order_year"] = d.dt.year
        X["order_month"] = d.dt.month
        X["order_day"] = d.dt.day
        X["is_weekend"] = d.dt.weekday.isin([5, 6]).astype(int)

        if self._include_cross_border:
            X["is_cross_border"] = (X[self.customer_country_col] != X[self.order_country_col]).astype(int)

        if self.order_region_col not in X.columns or self.market_col not in X.columns:
            raise ValueError(f"Expected '{self.order_region_col}' and '{self.market_col}' columns.")

        origin = _safe_series(X[self.order_region_col]).astype(str).str.replace(" ", "_", regex=False)
        destination = _safe_series(X[self.market_col]).astype(str).str.replace(" ", "_", regex=False)
        X["shipping_route"] = origin + "_to_" + destination

        return X


class TargetMeanEncoderMulticlass(BaseEstimator, TransformerMixin):
    """
    Multiclass target-mean encoder (one feature per (col, class)).
    """

    def __init__(self, smoothing: float = 20.0) -> None:
        self.smoothing = float(smoothing)

        self.classes_: Optional[np.ndarray] = None
        self.global_priors_: Optional[np.ndarray] = None
        self.maps_: dict[str, pd.DataFrame] = {}
        self.cols_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Sequence) -> "TargetMeanEncoderMulticlass":
        X = X.copy()
        y_s = pd.Series(y)
        self.classes_ = np.array(sorted(y_s.dropna().unique().tolist()))
        if self.classes_.size == 0:
            raise ValueError("TargetMeanEncoderMulticlass: no classes found in y.")

        y_onehot = pd.get_dummies(y_s, dtype=float).reindex(columns=self.classes_, fill_value=0.0)
        self.global_priors_ = y_onehot.mean(axis=0).to_numpy()

        self.cols_ = list(X.columns)

        for c in self.cols_:
            key = _safe_series(X[c]).astype("object")
            stats = y_onehot.groupby(key).agg(["mean", "count"])

            means = stats.xs("mean", axis=1, level=1)  # shape: (n_cat, n_classes)
            counts = (
                stats.xs("count", axis=1, level=1)
                .iloc[:, 0]
                .to_numpy()
                .astype(float)
                .reshape(-1, 1)  # (n_cat, 1)
            )

            priors = np.tile(self.global_priors_, (len(means), 1))  # (n_cat, n_classes)

            numer = means.to_numpy() * counts + priors * self.smoothing
            denom = counts + self.smoothing
            smooth = pd.DataFrame(
                numer / denom,
                index=means.index,
                columns=means.columns,
            )
            self.maps_[c] = smooth

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None or self.global_priors_ is None:
            raise RuntimeError("TargetMeanEncoderMulticlass must be fitted before transform().")

        if self.cols_ is None:
            raise RuntimeError("TargetMeanEncoderMulticlass has not been fitted.")

        X = X.copy()
        out_cols: List[np.ndarray] = []

        for c in self.cols_:
            mapping = self.maps_[c]
            key = _safe_series(X[c]).astype("object")

            # Ensure a writable array: pandas may return a read-only view depending on backend.
            enc = np.array(mapping.reindex(key).to_numpy(), copy=True)
            nan_rows = np.isnan(enc).all(axis=1)
            if nan_rows.any():
                enc[nan_rows] = self.global_priors_
            out_cols.append(enc)

        return np.concatenate(out_cols, axis=1) if out_cols else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ANN001
        if self.classes_ is None:
            raise RuntimeError("TargetMeanEncoderMulticlass must be fitted before get_feature_names_out().")
        names: List[str] = []
        if self.cols_ is None:
            raise RuntimeError("TargetMeanEncoderMulticlass has not been fitted.")
        for c in self.cols_:
            for k in self.classes_:
                names.append(f"{c}__te__{k}")
        return np.array(names, dtype=object)


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    target_col: str = TARGET_COL_DEFAULT
    date_col: str = DATE_COL_DEFAULT

    low_card_cols: Tuple[str, ...] = (
        "Type",
        "Shipping Mode",
        "Customer Segment",
        "Department Name",
    )
    high_card_cols: Tuple[str, ...] = (
        "Order City",
        "Customer City",
        "Product Name",
        "Category Name",
        "shipping_route",
    )


def build_preprocessor(
    df_example_after_feat: pd.DataFrame,
    config: FeatureEngineeringConfig = FeatureEngineeringConfig(),
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
    - one-hot encodes low-cardinality cols
    - target-encodes high-cardinality cols (multiclass)
    - passes through remaining numeric columns
    """

    numeric_cols = df_example_after_feat.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != config.target_col]

    handled = set(config.low_card_cols) | set(config.high_card_cols)
    numeric_passthrough = [c for c in numeric_cols if c not in handled]

    return ColumnTransformer(
        transformers=[
            ("ohe_low", OneHotEncoder(handle_unknown="ignore"), list(config.low_card_cols)),
            ("te_high", TargetMeanEncoderMulticlass(), list(config.high_card_cols)),
            ("num", "passthrough", numeric_passthrough),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    df_example: pd.DataFrame,
    model,
    *,
    scaler=None,
    oversampler=None,
    config: FeatureEngineeringConfig = FeatureEngineeringConfig(),
) -> ImbPipeline:
    """
    Build a customizable pipeline:

    FeatureBuilder -> ColumnTransformer -> (optional scaler) -> (optional oversampler) -> model

    Parameters
    ----------
    df_example : pd.DataFrame
        Example features before FeatureBuilder (e.g. training dataframe without target).
    model : estimator
        Any sklearn-compatible estimator (XGBoost, LogisticRegression, RandomForest, etc.).
    scaler : transformer or None
        Any sklearn scaler (StandardScaler, RobustScaler, etc.), applied after the ColumnTransformer.
    oversampler : sampler or None
        Any imblearn oversampler (RandomOverSampler, SMOTE, etc.), applied after scaling and
        only during fit.
    """

    feat = FeatureBuilder(date_col=config.date_col)
    X_example_feat = feat.fit_transform(df_example.copy())

    pre = build_preprocessor(X_example_feat, config=config)

    steps = [("feat", feat), ("pre", pre)]

    if scaler is not None:
        steps.append(("scaler", scaler))

    if oversampler is not None:
        steps.append(("sampler", oversampler))

    steps.append(("model", model))

    return ImbPipeline(steps=steps)

