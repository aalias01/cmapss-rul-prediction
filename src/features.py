"""
Feature engineering pipeline for CMAPSS turbofan RUL prediction.

This module is the single source of truth for how raw sensor data becomes
model-ready features. It is used by both the training notebooks and the
FastAPI inference layer (api/predictor.py), ensuring that the features seen
at training time exactly match those produced at inference time.

Pipeline summary
----------------
1. load_raw()              — parse space-delimited .txt files, assign column names
2. add_rul()               — compute piecewise-linear RUL labels for training data
3. drop_constant_sensors() — remove near-zero-variance channels identified by EDA
4. add_rolling_features()  — per-engine rolling mean and std over a 30-cycle window
5. build_features()        — steps 3 + 4 in sequence (the standard call from notebooks)

Constants
---------
CONSTANT_SENSORS : list[str]
    Sensors confirmed by variance analysis (01_eda.ipynb) to carry no predictive
    signal in FD001. Cross-referenced against Saxena & Goebel (2008).
RUL_CLIP : int
    Piecewise-linear cap (125 cycles). Standard CMAPSS convention — see README
    for the physical and statistical justification.
ROLLING_WINDOW : int
    Window size (30 cycles) for rolling statistics. This is a chosen smoothing
    horizon, not a measured fouling period or a validated optimum.
"""

import pandas as pd

COLUMNS = [
    "unit", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]

# Sensors with near-zero variance across FD001 — confirmed by 01_eda.ipynb variance analysis.
# Keeping these would add dimensionality without predictive signal.
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                    "sensor_16", "sensor_18", "sensor_19"]

RUL_CLIP = 125       # Piecewise-linear RUL cap (standard CMAPSS convention)
ROLLING_WINDOW = 30  # Chosen smoothing horizon, not a measured fouling period


def load_raw(filepath: str) -> pd.DataFrame:
    """
    Parse a CMAPSS space-delimited text file and return a labelled DataFrame.

    The raw files (train_FD001.txt, test_FD001.txt, etc.) have no header row
    and use variable-width whitespace as the delimiter. Column names are
    assigned from the module-level COLUMNS list.

    Parameters
    ----------
    filepath : str
        Path to a CMAPSS .txt file (train, test, or RUL).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: unit, cycle, op_setting_1/2/3, sensor_1–sensor_21.
    """
    return pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMNS)


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and clip piecewise-linear RUL labels for training data.

    For each engine unit, RUL at cycle t is defined as:
        RUL(t) = min(max_cycle − t, RUL_CLIP)

    The clip at RUL_CLIP (125) implements the standard piecewise-linear
    assumption: engines with more than 125 cycles remaining are treated as
    equally healthy, preventing the loss function from being dominated by
    the long plateau of early-life rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training DataFrame with 'unit' and 'cycle' columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a new 'rul' column and the intermediate
        'max_cycle' column removed.
    """
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=RUL_CLIP)
    return df.drop(columns=["max_cycle"])


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the near-zero-variance sensor columns identified during EDA.

    Only drops columns that are actually present, so this function is safe
    to call on DataFrames that have already had some sensors removed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw or partially processed sensor columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with CONSTANT_SENSORS columns removed.
    """
    return df.drop(columns=[c for c in CONSTANT_SENSORS if c in df.columns])


def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Add per-engine rolling mean and standard deviation for every sensor column.

    Rolling statistics are computed within each engine unit (grouped transform),
    so they never bleed information across engine boundaries. `min_periods=1`
    ensures that early cycles — where the window is not yet full — still receive
    a feature value rather than NaN.

    The resulting feature names follow the pattern:
        <sensor_name>_mean<window>   (e.g. sensor_2_mean30)
        <sensor_name>_std<window>    (e.g. sensor_2_std30)

    These are the features used by the XGBoost model and replicated in
    api/predictor.py for inference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'unit' column and one or more 'sensor_*' columns.
        Constant sensors should be dropped before calling this function.
    window : int, optional
        Rolling window size in cycles. Default is ROLLING_WINDOW (30), a chosen
        smoothing horizon rather than a validated physical period.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 2 × len(sensor_cols) additional columns appended.
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    rolled_mean = (
        df.groupby("unit")[sensor_cols]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
        .add_suffix(f"_mean{window}")
    )
    rolled_std = (
        df.groupby("unit")[sensor_cols]
        .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        .add_suffix(f"_std{window}")
    )
    return pd.concat([df, rolled_mean, rolled_std], axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a raw sensor DataFrame.

    This is the standard entry point for notebook use. It combines
    drop_constant_sensors() and add_rolling_features() in the correct order.
    The resulting DataFrame retains all original columns plus the rolling
    feature columns, and is ready for get_feature_cols() in src/model.py.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sensor DataFrame produced by load_raw() (with or without 'rul').

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame ready for model training or evaluation.
    """
    df = drop_constant_sensors(df)
    df = add_rolling_features(df)
    return df
