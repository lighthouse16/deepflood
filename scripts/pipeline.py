"""Shared V3.1 preprocessing and sequence contract."""
from __future__ import annotations

import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 7
EXCLUDED_FEATURES = {"date", "gauge_id", "streamflow", "year", "month", "day", "day_of_year"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features using current weather and strictly past streamflow."""
    df = df.copy()
    if "date" not in df.columns:
        if {"year", "month", "day"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        else:
            raise ValueError("Input requires date or year/month/day columns")
    df["date"] = pd.to_datetime(df["date"])
    if "gauge_id" not in df.columns:
        df["gauge_id"] = "longdai"
    missing = {"prcp", "tmax", "tmin", "streamflow"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.sort_values(["gauge_id", "date"]).reset_index(drop=True)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    for window in [3, 7, 15, 30, 60]:
        df[f"prcp_rolling_{window}d"] = df.groupby("gauge_id")["prcp"].transform(lambda x: x.rolling(window, min_periods=1).sum())
    for window in [7, 15, 30]:
        grouped = df.groupby("gauge_id")["prcp"]
        df[f"prcp_mean_{window}d"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"prcp_max_{window}d"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).max())
    df["temp_range"] = df["tmax"] - df["tmin"]
    for window in [7, 15]:
        df[f"tmax_rolling_{window}d"] = df.groupby("gauge_id")["tmax"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"streamflow_lag_{lag}"] = df.groupby("gauge_id")["streamflow"].shift(lag)
    for window in [3, 7, 14]:
        grouped = df.groupby("gauge_id")["streamflow"]
        df[f"streamflow_mean_{window}d"] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df[f"streamflow_std_{window}d"] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
    df["streamflow_change"] = df.groupby("gauge_id")["streamflow_lag_1"].diff()
    df["prcp_change"] = df.groupby("gauge_id")["prcp"].diff()
    df["prcp_x_streamflow_lag1"] = df["prcp"] * df["streamflow_lag_1"]
    keep = ["gauge_id", "date", "dayl", "prcp", "srad", "tmax", "tmin", "vp", "streamflow",
            "month_sin", "month_cos", "day_sin", "day_cos", "temp_range"]
    keep += [column for column in df.columns if column.startswith((
        "prcp_rolling_", "prcp_mean_", "prcp_max_", "tmax_rolling_",
        "streamflow_lag_", "streamflow_mean_", "streamflow_std_",
        "streamflow_change", "prcp_change", "prcp_x_streamflow"))]
    return df[keep].dropna().reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in EXCLUDED_FEATURES]


def create_sequences(df: pd.DataFrame, feature_cols: list[str], target_col: str = "streamflow"):
    X, y, dates, raw_targets = [], [], [], []
    for _, group in df.groupby("gauge_id"):
        group = group.sort_values("date").reset_index(drop=True)
        values, targets = group[feature_cols].to_numpy(), group[target_col].to_numpy()
        for end in range(SEQUENCE_LENGTH - 1, len(group)):
            X.append(values[end - SEQUENCE_LENGTH + 1:end + 1])
            y.append(targets[end]); dates.append(group.loc[end, "date"]); raw_targets.append(targets[end])
    return np.asarray(X), np.asarray(y), np.asarray(dates), np.asarray(raw_targets)
