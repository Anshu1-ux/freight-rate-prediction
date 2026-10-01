"""Cleaning + feature engineering shared by training, validation and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_WEIGHT = 47_500.0  # observed hard cap (legal-ish max payload) in the data


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data-quality issues. Uses no labels, so it is safe on validation data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1) negative weights: sign-flip errors (their magnitude matches the valid distribution)
    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight"] = df["weight"].abs()

    # 2) weight is hard-capped at 47,500 -> flag the censored rows, keep the value
    df["weight_capped"] = (df["weight"] >= MAX_WEIGHT).astype(int)

    # 3) missing weight: flag; LightGBM handles NaN natively
    df["weight_missing"] = df["weight"].isna().astype(int)

    # 4) missing market_index: fill with that day's median across loads (index is a daily
    #    market signal, within-day std ~0.025). Only same-file, same-day features used.
    df["market_index_missing"] = df["market_index"].isna().astype(int)
    day_med = df.groupby("date")["market_index"].transform("median")
    df["market_index"] = df["market_index"].fillna(day_med)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_distance"] = np.log(df["distance"])
    df["hav_miles"] = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    df["dist_ratio"] = df["distance"] / df["hav_miles"]
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear
    df["dom"] = df["date"].dt.day
    df["weight_frac"] = df["weight"] / MAX_WEIGHT
    df["heavy"] = (df["weight"] > 40_000).astype(float).where(df["weight"].notna())
    df["short_haul"] = (df["distance"] < 250).astype(int)
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]
    df["equipment"] = pd.Categorical(df["equipment"], categories=["Dry Van", "Flatbed", "Reefer"])
    return df


NUMERIC = [
    "log_distance", "distance", "hav_miles", "dist_ratio", "weight", "weight_frac", "heavy",
    "market_index", "quote_signal", "dow", "is_weekend", "month", "doy", "dom", "short_haul",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "lat_diff", "lon_diff",
    "weight_was_negative", "weight_capped", "weight_missing", "market_index_missing",
]
CATEGORICAL = ["equipment"]
FEATURES = NUMERIC + CATEGORICAL


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(clean(df))
