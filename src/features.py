"""
Feature engineering for CMAPSS turbofan RUL prediction.

Transforms raw sensor readings into predictive features using
rolling statistics and domain-informed degradation indicators.
"""

import pandas as pd
import numpy as np


COLUMNS = [
    "unit", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]

# Sensors with near-zero variance across FD001 — drop before modeling
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                    "sensor_16", "sensor_18", "sensor_19"]

RUL_CLIP = 125  # Piecewise-linear RUL cap (standard CMAPSS convention)
ROLLING_WINDOW = 30


def load_raw(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMNS)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and clip RUL for training data."""
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=RUL_CLIP)
    return df.drop(columns=["max_cycle"])


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in CONSTANT_SENSORS if c in df.columns])


def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add per-unit rolling mean and std for each sensor."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    rolled = (
        df.groupby("unit")[sensor_cols]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
        .add_suffix(f"_mean{window}")
    )
    rolled_std = (
        df.groupby("unit")[sensor_cols]
        .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        .add_suffix(f"_std{window}")
    )
    return pd.concat([df, rolled, rolled_std], axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_constant_sensors(df)
    df = add_rolling_features(df)
    return df
