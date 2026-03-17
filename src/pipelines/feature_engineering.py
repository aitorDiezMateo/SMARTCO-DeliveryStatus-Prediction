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
    Adds calendar features, cross-border flag, shipping route,
    and frequency-encoded counts for high-cardinality columns.

    Frequency maps are learned in fit() from training data only (CV-safe).
    """

    def __init__(
        self,
        date_col: str = DATE_COL_DEFAULT,
        customer_country_col: str = "Customer Country",
        order_country_col: str = "Order Country",
        order_region_col: str = "Order Region",
        market_col: str = "Market",
        freq_encode_cols: list | None = None,
    ) -> None:
        self.date_col = date_col
        self.customer_country_col = customer_country_col
        self.order_country_col = order_country_col
        self.order_region_col = order_region_col
        self.market_col = market_col
        self.freq_encode_cols = freq_encode_cols
        self._include_cross_border: bool = True
        self._freq_maps: dict[str, pd.Series] = {}

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ANN001
        X = X.copy()
        if self.customer_country_col not in X.columns or self.order_country_col not in X.columns:
            raise ValueError(
                f"Expected '{self.customer_country_col}' and '{self.order_country_col}' columns."
            )
        cross = (X[self.customer_country_col] != X[self.order_country_col]).astype(float)
        uniq = pd.unique(cross.dropna())
        self._include_cross_border = not (len(uniq) == 1 and float(uniq[0]) == 1.0)

        self._freq_maps = {}
        for col in (self.freq_encode_cols or []):
            if col in X.columns:
                self._freq_maps[col] = X[col].value_counts()

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
        X["day_of_week"] = d.dt.dayofweek
        X["week_of_year"] = d.dt.isocalendar().week.values.astype(int)
        X["quarter"] = d.dt.quarter
        X["is_month_start"] = d.dt.is_month_start.astype(int)
        X["is_month_end"] = d.dt.is_month_end.astype(int)

        if self._include_cross_border:
            X["is_cross_border"] = (X[self.customer_country_col] != X[self.order_country_col]).astype(int)

        if self.order_region_col not in X.columns or self.market_col not in X.columns:
            raise ValueError(f"Expected '{self.order_region_col}' and '{self.market_col}' columns.")

        origin = _safe_series(X[self.order_region_col]).astype(str).str.replace(" ", "_", regex=False)
        destination = _safe_series(X[self.market_col]).astype(str).str.replace(" ", "_", regex=False)
        X["shipping_route"] = origin + "_to_" + destination

        for col, counts in self._freq_maps.items():
            if col in X.columns:
                X[f"{col}_freq"] = X[col].map(counts).fillna(1).astype(int)

        return X


class HistoricalTargetStats(BaseEstimator, TransformerMixin):
    """
    Time-aware, CV-safe historical target statistics by group.

    For training data (fit_transform), it creates *past-only* features per row by:
    - sorting by date
    - computing cumulative class counts per group, shifted by 1 (excluding current row)
    - converting counts to smoothed probabilities per class
    - adding a past count feature per group

    For inference data (transform), it maps each group to the final (end-of-train)
    posterior probabilities and total counts learned in fit(). This is appropriate
    when inference rows occur after the training window (e.g., time series CV valid
    folds and the global test set).
    """

    def __init__(
        self,
        *,
        date_col: str = DATE_COL_DEFAULT,
        group_cols: Tuple[str, ...] = ("shipping_route", "Market"),
        smoothing: float = 20.0,
        classes: Optional[Sequence] = None,
    ) -> None:
        self.date_col = date_col
        self.group_cols = tuple(group_cols)
        self.smoothing = float(smoothing)

        self.classes_: Optional[np.ndarray] = np.array(list(classes)) if classes is not None else None
        self.global_priors_: Optional[np.ndarray] = None
        self._group_posteriors: dict[str, pd.DataFrame] = {}
        self._group_counts: dict[str, pd.Series] = {}

    def add_placeholder_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of X with the historical feature columns added (filled with 0).
        This is used to build downstream preprocessors without fitting on fake labels.
        """
        if self.classes_ is None:
            raise ValueError("HistoricalTargetStats.add_placeholder_columns requires `classes` to be provided.")

        X_out = X.copy()
        for col in self.group_cols:
            if col not in X_out.columns:
                continue
            for cls in self.classes_:
                X_out[f"{col}__hist_p__{cls}"] = 0.0
            X_out[f"{col}__hist_count"] = 0
        return X_out

    def fit(self, X: pd.DataFrame, y: Sequence):  # noqa: ANN001
        X = X.copy()
        y_s = pd.Series(y)
        if self.classes_ is None:
            self.classes_ = np.array(sorted(y_s.dropna().unique().tolist()))
        if self.classes_.size == 0:
            raise ValueError("HistoricalTargetStats: no classes found in y.")

        y_onehot = pd.get_dummies(y_s, dtype=float).reindex(columns=self.classes_, fill_value=0.0)
        self.global_priors_ = y_onehot.mean(axis=0).to_numpy()

        # Learn end-of-train posteriors for each group for transform() usage.
        self._group_posteriors = {}
        self._group_counts = {}

        for col in self.group_cols:
            if col not in X.columns:
                continue
            key = _safe_series(X[col]).astype("object")
            stats = y_onehot.groupby(key).agg(["sum", "count"])
            sums = stats.xs("sum", axis=1, level=1)  # (n_groups, n_classes)
            counts = (
                stats.xs("count", axis=1, level=1)
                .iloc[:, 0]
                .to_numpy()
                .astype(float)
                .reshape(-1, 1)
            )

            priors = np.tile(self.global_priors_, (len(sums), 1))
            numer = sums.to_numpy() + priors * self.smoothing
            denom = counts + self.smoothing
            post = pd.DataFrame(
                numer / denom,
                index=sums.index,
                columns=sums.columns,
            )
            self._group_posteriors[col] = post
            self._group_counts[col] = pd.Series(
                counts.reshape(-1).astype(int),
                index=sums.index,
                name=f"{col}__hist_count",
            )

        return self

    def fit_transform(self, X: pd.DataFrame, y: Sequence, **fit_params):  # noqa: ANN001
        self.fit(X, y)

        X_out = X.copy()
        if self.classes_ is None or self.global_priors_ is None:
            raise RuntimeError("HistoricalTargetStats: fit did not initialise state.")

        if self.date_col not in X_out.columns:
            raise ValueError(f"HistoricalTargetStats: expected date column '{self.date_col}' not found.")

        d = pd.to_datetime(X_out[self.date_col], errors="coerce")
        if d.isna().any():
            raise ValueError(
                f"HistoricalTargetStats: date column '{self.date_col}' contains invalid values."
            )

        order = np.argsort(d.to_numpy())
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        y_s = pd.Series(y).reset_index(drop=True)
        y_onehot = pd.get_dummies(y_s, dtype=float).reindex(columns=self.classes_, fill_value=0.0)

        X_sorted = X_out.iloc[order].reset_index(drop=True)
        y_sorted = y_onehot.iloc[order].reset_index(drop=True)

        for col in self.group_cols:
            if col not in X_sorted.columns:
                continue

            key = _safe_series(X_sorted[col]).astype("object")
            # Cumulative sums and counts, shifted to exclude current row.
            cum_sums = y_sorted.groupby(key, sort=False).cumsum().shift(1, fill_value=0.0)
            cum_counts = y_sorted.groupby(key, sort=False).cumcount()  # 0 for first row

            priors = np.tile(self.global_priors_, (len(X_sorted), 1))
            numer = cum_sums.to_numpy() + priors * self.smoothing
            denom = (cum_counts.to_numpy().astype(float).reshape(-1, 1) + self.smoothing)
            probs = numer / denom

            for j, cls in enumerate(self.classes_):
                X_sorted[f"{col}__hist_p__{cls}"] = probs[:, j]
            X_sorted[f"{col}__hist_count"] = cum_counts.to_numpy().astype(int)

        # Restore original order
        X_restored = X_sorted.iloc[inv_order].set_index(X_out.index)
        return X_restored

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.classes_ is None or self.global_priors_ is None:
            raise RuntimeError("HistoricalTargetStats must be fitted before transform().")

        X_out = X.copy()

        for col in self.group_cols:
            if col not in X_out.columns:
                continue

            post = self._group_posteriors.get(col)
            counts = self._group_counts.get(col)

            key = _safe_series(X_out[col]).astype("object")

            if post is None or counts is None:
                # Fall back to global priors / zeros if the group col wasn't present in fit.
                for cls in self.classes_:
                    X_out[f"{col}__hist_p__{cls}"] = float(self.global_priors_[int(np.where(self.classes_ == cls)[0][0])])
                X_out[f"{col}__hist_count"] = 0
                continue

            mapped = post.reindex(key).to_numpy()
            mapped = np.array(mapped, copy=True)
            nan_rows = np.isnan(mapped).all(axis=1)
            if nan_rows.any():
                mapped[nan_rows] = self.global_priors_

            for j, cls in enumerate(self.classes_):
                X_out[f"{col}__hist_p__{cls}"] = mapped[:, j]

            cnt = counts.reindex(key).fillna(0).astype(int).to_numpy()
            X_out[f"{col}__hist_count"] = cnt

        return X_out


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
        "Market",
    )
    high_card_cols: Tuple[str, ...] = (
        "Order City",
        "Customer City",
        "Product Name",
        "Category Name",
        "Order Region",
        "shipping_route",
    )
    freq_encode_cols: Tuple[str, ...] = (
        "Order City",
        "Customer City",
        "Product Name",
        "Category Name",
        "Order Region",
        "Market",
    )


def build_preprocessor(
    df_example_after_feat: pd.DataFrame,
    config: FeatureEngineeringConfig = FeatureEngineeringConfig(),
    smoothing: float = 20.0,
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
            ("te_high", TargetMeanEncoderMulticlass(smoothing=smoothing), list(config.high_card_cols)),
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
    smoothing: float = 20.0,
    classes: Optional[Sequence] = None,
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
    smoothing : float
        Smoothing strength for the target mean encoder (higher = more regularisation).
    classes : Sequence or None
        Ordered list of target classes (e.g. [0, 1, 2]). Required to create stable
        historical-stat columns for multiclass without fitting on dummy labels.
    """

    feat = FeatureBuilder(
        date_col=config.date_col,
        freq_encode_cols=list(config.freq_encode_cols),
    )
    X_example_feat = feat.fit_transform(df_example.copy())

    hist = HistoricalTargetStats(
        date_col=config.date_col,
        group_cols=("shipping_route", "Market"),
        smoothing=smoothing,
        classes=classes,
    )
    # Ensure historical feature columns exist when building the ColumnTransformer.
    # We add placeholders here; real values are computed during fit/transform.
    if hist.classes_ is None:
        # Fallback: assume binary if classes weren't provided by the caller/model.
        hist = HistoricalTargetStats(
            date_col=config.date_col,
            group_cols=("shipping_route", "Market"),
            smoothing=smoothing,
            classes=[0, 1],
        )
    X_example_feat_hist = hist.add_placeholder_columns(X_example_feat.copy())

    pre = build_preprocessor(X_example_feat_hist, config=config, smoothing=smoothing)

    steps = [("feat", feat), ("hist", hist), ("pre", pre)]

    if scaler is not None:
        steps.append(("scaler", scaler))

    if oversampler is not None:
        steps.append(("sampler", oversampler))

    steps.append(("model", model))

    return ImbPipeline(steps=steps)
