"""
Model training and evaluation for CMAPSS RUL prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


FEATURE_COLS = None  # Set dynamically after feature engineering
TARGET = "rul"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"unit", "cycle", "rul", "op_setting_1", "op_setting_2", "op_setting_3"}
    return [c for c in df.columns if c not in exclude]


def train_test_split_by_unit(df: pd.DataFrame, test_units: list[int]):
    """
    Split by engine unit — NEVER split randomly on time-series sensor data.
    Data from the same engine must stay in the same split to prevent leakage.
    """
    train = df[~df["unit"].isin(test_units)]
    test = df[df["unit"].isin(test_units)]
    return train, test


def train_xgb(X_train, y_train) -> XGBRegressor:
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


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test).clip(0, 125)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"rmse": round(rmse, 3), "predictions": preds}
