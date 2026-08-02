"""Reusable feature engineering and model helpers for demand forecasting."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


KEYS = ["Date", "Store", "Product"]
LAGS = list(range(1, 15)) + [21, 28]

CATEGORICAL_DEFAULTS = {
    "Type": "unknown",
    "Weekday": "unknown",
    "DayType": "unknown",
    "Weather": "unknown",
    "PromotionType": "none",
}

NUMERIC_DEFAULTS = {
    "TemperatureMin": np.nan,
    "TemperatureMax": np.nan,
    "TemperatureAvg": np.nan,
    "ShopOpen": 1,
    "IsEventDay": 0,
    "DiscountRate": 0,
    "FreeChouxPerCustomer": 0,
    "MaxPromotionCustomers": 0,
    "MinimumPurchaseUnits": 0,
    "AnyProductQualifies": 0,
}

CATEGORICAL_FEATURES = [
    "Store",
    "Product",
    "Type",
    "Weekday",
    "DayType",
    "Weather",
    "PromotionType",
]

BASE_NUMERIC_FEATURES = [
    "DayIndex",
    "DayOfWeekSin",
    "DayOfWeekCos",
    "DayOfYearSin",
    "DayOfYearCos",
    "IsWeekend",
    "IsHolidayOrWeekend",
    "TemperatureMin",
    "TemperatureMax",
    "TemperatureAvg",
    "ShopOpen",
    "IsEventDay",
    "DiscountRate",
    "FreeChouxPerCustomer",
    "MaxPromotionCustomers",
    "MinimumPurchaseUnits",
    "AnyProductQualifies",
    *[f"Lag{lag}" for lag in LAGS],
    "Recent3Mean",
    "Recent7Mean",
    "Recent7Median",
    "PreviousWeekMean",
    "SameWeekdayMean",
    "RecentTrend",
    "Lag7Change",
    "SeriesMeanBefore",
    "SeriesStdBefore",
    "StoreTotalLag1",
    "StoreTotalLag7",
    "ProductTotalLag1",
    "ProductTotalLag7",
    "AllTotalLag1",
    "AllTotalLag7",
]


def _normalise_optional_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add optional columns with stable types before feature creation."""
    result = data.copy()
    result["Date"] = pd.to_datetime(result["Date"]).dt.normalize()

    for column, default in CATEGORICAL_DEFAULTS.items():
        if column not in result.columns:
            result[column] = default
        result[column] = result[column].astype("string").fillna(default).astype(str)

    for column, default in NUMERIC_DEFAULTS.items():
        if column not in result.columns:
            result[column] = default
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def _add_exact_date_lags(result: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Add demand from exact earlier dates without using future rows."""
    for lag in LAGS:
        lagged = history.copy()
        lagged["Date"] += pd.Timedelta(days=lag)
        lagged = lagged.rename(columns={"Demand": f"Lag{lag}"})
        result = result.merge(
            lagged,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    return result


def _add_aggregate_lags(result: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Add prior-day and prior-week totals at store, product, and all-store level."""
    levels = {
        "Store": ["Store"],
        "Product": ["Product"],
        "All": [],
    }

    for level_name, group_columns in levels.items():
        group_keys = ["Date", *group_columns]
        totals = (
            history.groupby(group_keys, as_index=False)["Demand"]
            .sum()
            .rename(columns={"Demand": f"{level_name}Total"})
        )
        for lag in [1, 7]:
            lagged = totals.copy()
            lagged["Date"] += pd.Timedelta(days=lag)
            lagged = lagged.rename(
                columns={f"{level_name}Total": f"{level_name}TotalLag{lag}"}
            )
            result = result.merge(
                lagged,
                on=group_keys,
                how="left",
                validate="many_to_one",
            )
    return result


def _add_history_statistics(result: pd.DataFrame) -> pd.DataFrame:
    """Add expanding statistics that use only earlier observations."""
    result = result.sort_values(["Store", "Product", "Date"]).reset_index(drop=True)
    grouped = result.groupby(["Store", "Product"], sort=False)["Demand"]
    result["SeriesMeanBefore"] = grouped.transform(
        lambda values: values.shift(1).expanding().mean()
    )
    result["SeriesStdBefore"] = grouped.transform(
        lambda values: values.shift(1).expanding().std()
    )
    return result


def _add_calendar_cycles(result: pd.DataFrame, origin_date: pd.Timestamp) -> None:
    """Add smooth weekly/yearly cycles and a linear time index in place."""
    day_of_week = result["Date"].dt.dayofweek
    day_of_year = result["Date"].dt.dayofyear

    result["DayIndex"] = (result["Date"] - origin_date).dt.days
    result["DayOfWeekSin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["DayOfWeekCos"] = np.cos(2 * np.pi * day_of_week / 7)
    result["DayOfYearSin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["DayOfYearCos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    result["IsWeekend"] = day_of_week.isin([5, 6]).astype(int)
    result["IsHolidayOrWeekend"] = result["DayType"].ne("平日").astype(int)


def _add_recent_demand_features(result: pd.DataFrame) -> None:
    """Summarise recent lags into stable short-term demand signals in place."""
    result["Recent3Mean"] = result[[f"Lag{lag}" for lag in range(1, 4)]].mean(axis=1)
    result["Recent7Mean"] = result[[f"Lag{lag}" for lag in range(1, 8)]].mean(axis=1)
    result["Recent7Median"] = result[[f"Lag{lag}" for lag in range(1, 8)]].median(axis=1)
    result["PreviousWeekMean"] = result[[f"Lag{lag}" for lag in range(8, 15)]].mean(axis=1)
    result["SameWeekdayMean"] = result[["Lag7", "Lag14", "Lag21", "Lag28"]].mean(axis=1)
    result["RecentTrend"] = result["Recent3Mean"] - result["PreviousWeekMean"]
    result["Lag7Change"] = result["Lag1"] - result["Lag7"]


def build_features(data: pd.DataFrame, origin_date: pd.Timestamp) -> pd.DataFrame:
    """Build leakage-safe demand features for historical and future rows."""
    result = _normalise_optional_columns(data)
    history = result.loc[result["Demand"].notna(), KEYS + ["Demand"]].copy()
    result = _add_exact_date_lags(result, history)
    result = _add_aggregate_lags(result, history)
    result = _add_history_statistics(result)
    _add_calendar_cycles(result, pd.Timestamp(origin_date))
    _add_recent_demand_features(result)
    return result


def select_model_features(model_data: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return categorical, usable numeric, and combined feature lists."""
    numeric_features = [
        column
        for column in BASE_NUMERIC_FEATURES
        if column in model_data.columns and model_data[column].notna().any()
    ]
    return (
        CATEGORICAL_FEATURES.copy(),
        numeric_features,
        CATEGORICAL_FEATURES + numeric_features,
    )


def make_model_specs() -> dict[str, object]:
    """Return the three regressors used by the ensemble."""
    return {
        "LightGBM Poisson": LGBMRegressor(
            objective="poisson",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=15,
            subsample=0.90,
            colsample_bytree=0.80,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        "Histogram GB Poisson": HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=350,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        ),
        "Histogram GB absolute": HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=350,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def make_pipeline(
    estimator: object,
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Build preprocessing and model steps as one reusable pipeline."""
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn before 1.2
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clone(estimator)),
        ]
    )


def calculate_metrics(actual: Iterable[float], prediction: Iterable[float]) -> dict[str, float]:
    """Calculate unit-scale forecast metrics."""
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    absolute_error = np.abs(actual_array - prediction_array)
    return {
        "MAE": float(mean_absolute_error(actual_array, prediction_array)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_array, prediction_array))),
        "WAPE": (
            float(absolute_error.sum() / actual_array.sum())
            if actual_array.sum() > 0
            else np.nan
        ),
        "Bias": float(np.mean(prediction_array - actual_array)),
    }


def make_rolling_folds(
    model_data: pd.DataFrame,
    *,
    fold_days: int = 7,
    max_folds: int = 4,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]]:
    """Create chronological folds using dates with near-complete product coverage."""
    daily_counts = model_data.groupby("Date").size().sort_index()
    minimum_rows = max(1, int(daily_counts.median() * 0.80))
    complete_dates = daily_counts[daily_counts >= minimum_rows].index.sort_values()

    number_of_folds = min(max_folds, max(2, len(complete_dates) // fold_days - 2))
    last_complete_date = complete_dates.max()
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]] = []

    for fold_number in range(number_of_folds, 0, -1):
        fold_start = last_complete_date - pd.Timedelta(
            days=fold_days * fold_number - 1
        )
        fold_end = fold_start + pd.Timedelta(days=fold_days - 1)
        validation_dates = complete_dates[
            (complete_dates >= fold_start) & (complete_dates <= fold_end)
        ]
        if len(validation_dates) >= 5:
            folds.append((fold_start, fold_end, validation_dates))

    if len(folds) < 2:
        raise ValueError("Not enough complete dates for rolling validation.")
    return folds
