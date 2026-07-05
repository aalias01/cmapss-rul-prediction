"""
XGBoost training and evaluation for CMAPSS turbofan RUL prediction.

This module handles everything between feature-engineered data and a saved
model artifact: selecting the right feature columns, splitting by engine unit
to prevent temporal leakage, training an XGBoost regressor, and evaluating
it with RMSE.

The trained model is saved to models/xgb_rul.joblib by the training notebook
(03_modeling.ipynb) and loaded at API startup by api/predictor.py.

Temporal leakage note
---------------------
Random train/test splits are incorrect for this dataset. Each engine produces
a time-ordered sequence of rows; splitting randomly would place early and late
cycles from the same engine in both train and test, allowing the model to
memorise per-engine drift rather than learning a generalised degradation
pattern. train_test_split_by_unit() enforces strict engine-level separation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

TARGET = "rul"

# Columns excluded from the feature matrix regardless of what is present.
# Operating settings are excluded because FD001 has a single operating
# condition (op_setting_3 is always 100.0); including them adds noise.
_NON_FEATURE_COLS = {"unit", "cycle", "rul",
                     "op_setting_1", "op_setting_2", "op_setting_3"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the list of column names to use as model features.

    Excludes metadata columns (unit, cycle), the target (rul), and operating
    setting columns (constant in FD001). Everything else — raw sensor readings
    and rolling statistics — is included.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame produced by src/features.build_features().

    Returns
    -------
    list[str]
        Ordered list of feature column names, matching the column order in df.
    """
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def train_test_split_by_unit(
    df: pd.DataFrame, test_units: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset by engine unit, keeping all cycles of each engine together.

    This is the only correct train/test split strategy for time-series sensor
    data. Splitting randomly across rows would leak information between train
    and test: cycles from the same engine in both splits allow the model to
    memorise per-engine behaviour rather than generalising.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature-engineered DataFrame with a 'unit' column.
    test_units : list[int]
        Engine unit IDs to hold out as the test split.

    Returns
    -------
    train : pd.DataFrame
        All rows belonging to engines NOT in test_units.
    test : pd.DataFrame
        All rows belonging to engines in test_units.
    """
    train = df[~df["unit"].isin(test_units)]
    test  = df[df["unit"].isin(test_units)]
    return train, test


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Train an XGBoost regressor on the provided feature matrix and RUL targets.

    Hyperparameters were chosen to balance model capacity against overfitting
    on 100 training engines: moderate depth (6), low learning rate (0.05),
    and row/column subsampling (0.8 each) for regularisation.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix from get_feature_cols().
    y_train : pd.Series
        Clipped RUL labels (0–125) from src/features.add_rul().

    Returns
    -------
    XGBRegressor
        Fitted XGBoost model. Save with joblib.dump() for API use.
    """
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a trained model and return RMSE along with the raw predictions.

    Predictions are clipped to [0, 125] before scoring — negative RUL values
    are physically meaningless, and values above 125 exceed the label range.

    Parameters
    ----------
    model : XGBRegressor
        Trained model returned by train_xgb().
    X_test : pd.DataFrame
        Feature matrix for the held-out test split.
    y_test : pd.Series
        True RUL labels for the test split.

    Returns
    -------
    dict
        {
          "rmse": float — root mean squared error (rounded to 3 decimal places),
          "predictions": np.ndarray — clipped predicted RUL values
        }
    """
    preds = model.predict(X_test).clip(0, 125)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"rmse": round(rmse, 3), "predictions": preds}
