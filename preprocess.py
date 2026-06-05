from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# NYC bounding box (Kaggle / TLC filters)
LAT_MIN, LAT_MAX = 40.477399, 40.917577
LON_MIN, LON_MAX = -74.259090, -73.700272

LAT_KM_PER_DEG = 111
LON_KM_PER_DEG = 85  # approx km per degree longitude at NYC latitude

PICKUP_DATE_MIN = pd.Timestamp("2016-01-01")
PICKUP_DATE_MAX = pd.Timestamp("2016-06-30 23:59:59")

MIN_DURATION_SEC = 60
MAX_DURATION_SEC = 24 * 3600
MIN_PASSENGERS = 1
MAX_PASSENGERS = 6
MIN_SPEED_KMH = 1
MAX_SPEED_KMH = 120

DEFAULT_VAL_START = "2016-06-13"  # last ~2.5 weeks of training data


def manhattan_km(df: pd.DataFrame) -> pd.Series:
    """Approximate route distance in km (latitude/longitude Manhattan metric)."""
    return (
        np.abs(df["dropoff_latitude"] - df["pickup_latitude"]) * LAT_KM_PER_DEG
        + np.abs(df["dropoff_longitude"] - df["pickup_longitude"]) * LON_KM_PER_DEG
    )


def add_manhattan_km(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["manhattan_km"] = manhattan_km(out)
    return out


def haversine_km(df: pd.DataFrame) -> pd.Series:
    """Great-circle distance in km between pickup and dropoff coordinates."""
    lat1 = np.radians(df["pickup_latitude"].to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].to_numpy())
    dlat = np.radians((df["dropoff_latitude"] - df["pickup_latitude"]).to_numpy())
    dlon = np.radians((df["dropoff_longitude"] - df["pickup_longitude"]).to_numpy())
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return pd.Series(6371.0 * 2 * np.arcsin(np.sqrt(a)), index=df.index)


def add_haversine_km(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["haversine_km"] = haversine_km(out)
    return out


def _coords_in_nyc(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in (
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ):
        if "latitude" in col:
            mask &= df[col].between(LAT_MIN, LAT_MAX)
        else:
            mask &= df[col].between(LON_MIN, LON_MAX)
    return mask


def compute_clean_mask(df: pd.DataFrame, *, require_target: bool = True) -> pd.Series:
    """
    Boolean mask for rows that pass data-quality rules from EDA.

    When require_target is False (e.g. test.csv), duration and speed rules are skipped.
    """
    work = df
    if "pickup_datetime" not in work.columns or work["pickup_datetime"].dtype == object:
        pickup_dt = pd.to_datetime(work["pickup_datetime"])
    else:
        pickup_dt = work["pickup_datetime"]

    in_nyc = _coords_in_nyc(work)
    valid_passengers = work["passenger_count"].between(MIN_PASSENGERS, MAX_PASSENGERS)
    valid_dates = pickup_dt.between(PICKUP_DATE_MIN, PICKUP_DATE_MAX)

    if "manhattan_km" in work.columns:
        km = work["manhattan_km"]
    else:
        km = manhattan_km(work)
    positive_distance = km > 0

    if require_target and "trip_duration" in work.columns:
        valid_duration = work["trip_duration"].between(MIN_DURATION_SEC, MAX_DURATION_SEC)
        speed_kmh = km / (work["trip_duration"] / 3600)
        valid_speed = speed_kmh.between(MIN_SPEED_KMH, MAX_SPEED_KMH)
    else:
        valid_duration = pd.Series(True, index=work.index)
        valid_speed = pd.Series(True, index=work.index)

    return (
        in_nyc
        & valid_passengers
        & valid_duration
        & valid_dates
        & positive_distance
        & valid_speed
    )


def quality_check_summary(df: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Per-rule failure counts (matches EDA quality_checks table)."""
    work = add_manhattan_km(df) if "manhattan_km" not in df.columns else df.copy()
    if work["pickup_datetime"].dtype == object:
        work["pickup_datetime"] = pd.to_datetime(work["pickup_datetime"])

    in_nyc = _coords_in_nyc(work)
    valid_passengers = work["passenger_count"].between(MIN_PASSENGERS, MAX_PASSENGERS)
    valid_dates = work["pickup_datetime"].between(PICKUP_DATE_MIN, PICKUP_DATE_MAX)
    positive_distance = work["manhattan_km"] > 0

    rules = [
        ("coordinates in NYC bbox", ~in_nyc),
        ("passenger_count in [1, 6]", ~valid_passengers),
        ("pickup in Jan–Jun 2016", ~valid_dates),
        ("manhattan_km > 0", ~positive_distance),
    ]

    if require_target and "trip_duration" in work.columns:
        valid_duration = work["trip_duration"].between(MIN_DURATION_SEC, MAX_DURATION_SEC)
        speed_kmh = work["manhattan_km"] / (work["trip_duration"] / 3600)
        valid_speed = speed_kmh.between(MIN_SPEED_KMH, MAX_SPEED_KMH)
        rules.extend(
            [
                ("trip_duration in [60s, 24h]", ~valid_duration),
                ("implied speed in [1, 120] km/h", ~valid_speed),
            ]
        )

    summary = pd.DataFrame(
        {
            "rule": [name for name, _ in rules],
            "rows_failing": [mask.sum() for _, mask in rules],
        }
    )
    summary["pct_of_data"] = (100 * summary["rows_failing"] / len(work)).round(3)
    return summary


def load_train(path: str | Path = "data/train.csv") -> pd.DataFrame:
    """Load train CSV and drop leakage / ID columns."""
    df = pd.read_csv(path)
    drop_cols = [c for c in ("id", "dropoff_datetime") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    return df


def load_test(path: str | Path = "data/test.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [c for c in ("id",) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    return df


def clean_dataframe(df: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Return a copy of rows passing all cleaning rules, with manhattan_km added."""
    work = add_manhattan_km(df)
    mask = compute_clean_mask(work, require_target=require_target)
    return work.loc[mask].copy()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out["pickup_datetime"].dtype == object:
        out["pickup_datetime"] = pd.to_datetime(out["pickup_datetime"])
    hour = out["pickup_datetime"].dt.hour
    dow = out["pickup_datetime"].dt.dayofweek
    out["pickup_hour"] = hour
    out["pickup_dow"] = dow
    out["pickup_month"] = out["pickup_datetime"].dt.month
    out["is_weekend"] = (dow >= 5).astype(int)
    # Weekday morning (7–9) and evening (16–19) rush windows
    rush_hours = {7, 8, 9, 16, 17, 18, 19}
    out["is_rush_hour"] = ((dow < 5) & hour.isin(rush_hours)).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def add_log_target(df: pd.DataFrame, col: str = "trip_duration") -> pd.DataFrame:
    out = df.copy()
    out["log_trip_duration"] = np.log1p(out[col])
    return out


METADATA_FEATURES = ["vendor_id", "passenger_count", "store_and_fwd_flag"]
TIME_FEATURES = [
    "pickup_hour",
    "pickup_dow",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "hour_sin",
    "hour_cos",
]
DISTANCE_FEATURES = ["manhattan_km", "haversine_km"]
COORDINATE_FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
FULL_FEATURES = METADATA_FEATURES + TIME_FEATURES + DISTANCE_FEATURES + COORDINATE_FEATURES

FEATURE_GROUPS: dict[str, list[str]] = {
    "metadata_only": METADATA_FEATURES,
    "time_only": TIME_FEATURES,
    "distance_only": DISTANCE_FEATURES,
    "coordinates_only": COORDINATE_FEATURES,
    "distance_time": DISTANCE_FEATURES + TIME_FEATURES,
    "distance_coordinates": DISTANCE_FEATURES + COORDINATE_FEATURES,
    "full": FULL_FEATURES,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-trip feature matrix for tabular models.

    All columns are known at pickup time (no dropoff_datetime, trip_duration, or speed).
    """
    work = add_time_features(add_haversine_km(add_manhattan_km(df)))
    fwd = work["store_and_fwd_flag"].map({"N": 0, "Y": 1}).fillna(0).astype(int)
    features = {
        "manhattan_km": work["manhattan_km"],
        "haversine_km": work["haversine_km"],
        "pickup_hour": work["pickup_hour"],
        "pickup_dow": work["pickup_dow"],
        "pickup_month": work["pickup_month"],
        "is_weekend": work["is_weekend"],
        "is_rush_hour": work["is_rush_hour"],
        "hour_sin": work["hour_sin"],
        "hour_cos": work["hour_cos"],
        "passenger_count": work["passenger_count"],
        "vendor_id": work["vendor_id"],
        "store_and_fwd_flag": fwd,
        "pickup_longitude": work["pickup_longitude"],
        "pickup_latitude": work["pickup_latitude"],
        "dropoff_longitude": work["dropoff_longitude"],
        "dropoff_latitude": work["dropoff_latitude"],
    }
    return pd.DataFrame(features, index=work.index)[FULL_FEATURES]


def select_features(X: pd.DataFrame, group: str) -> pd.DataFrame:
    """Return a feature subset for ablation experiments."""
    if group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group {group!r}; choose from {list(FEATURE_GROUPS)}")
    cols = FEATURE_GROUPS[group]
    return X[cols]


def train_val_split_by_date(
    df: pd.DataFrame,
    val_start: str | pd.Timestamp = DEFAULT_VAL_START,
    datetime_col: str = "pickup_datetime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: train on pickups before val_start, validate on/after."""
    cutoff = pd.Timestamp(val_start)
    if df[datetime_col].dtype == object:
        pickup = pd.to_datetime(df[datetime_col])
    else:
        pickup = df[datetime_col]
    train = df.loc[pickup < cutoff].copy()
    val = df.loc[pickup >= cutoff].copy()
    return train, val


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_log_duration(y_true_sec: np.ndarray, y_pred_sec: np.ndarray) -> float:
    """RMSE on log1p(duration) — common Kaggle-style metric."""
    return rmse(np.log1p(y_true_sec), np.log1p(np.maximum(y_pred_sec, 0)))


def mae_seconds(y_true_sec: np.ndarray, y_pred_sec: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true_sec) - np.asarray(y_pred_sec))))


def evaluate_predictions(
    y_true_sec: np.ndarray,
    y_pred_sec: np.ndarray,
) -> dict[str, float]:
    y_pred_sec = np.maximum(np.asarray(y_pred_sec, dtype=float), 0)
    return {
        "rmse_log": rmse_log_duration(y_true_sec, y_pred_sec),
        "rmse_sec": rmse(y_true_sec, y_pred_sec),
        "mae_sec": mae_seconds(y_true_sec, y_pred_sec),
    }
