"""
Model loading and inference for the turbofan RUL prediction API.

This module is the bridge between the trained XGBoost artifact (produced by
03_modeling.ipynb and saved to models/xgb_rul.joblib) and the FastAPI routes
in api/main.py. Keeping inference logic here — separate from the route
definitions — keeps the routes thin and makes the prediction pipeline
independently testable.

Inference pipeline (per request)
---------------------------------
1. readings_to_dataframe()  — convert the JSON payload to a DataFrame and
                              apply the same rolling-feature transform used
                              at training time (src/features.add_rolling_features)
2. predict()                — select the most recent cycle, run XGBoost,
                              clip the output, compute SHAP values, and
                              return a structured PredictResponse

SHAP note
---------
SHAP values are computed with XGBoost's built-in exact TreeSHAP
(booster.predict(pred_contribs=True)) rather than the shap package's
TreeExplainer. The two are mathematically identical for XGBoost models,
but the shap package's loader cannot parse the base_score format
serialised by XGBoost >= 3.0 — and skipping the shap dependency keeps
the Render deployment image small.

Module-level caching
--------------------
_model and _booster are loaded once at server startup (via main.py's
lifespan handler) and reused for every subsequent request. This avoids the
~200 ms joblib initialisation cost on every call.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from api.schemas import PredictRequest, PredictResponse, ShapEntry

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_rul.joblib"

# Module-level model cache — populated by load_model() at server startup.
_model = None
_booster = None


def load_model() -> None:
    """
    Load the trained XGBoost model and cache its underlying Booster.

    Called once during server startup via FastAPI's lifespan handler. Raises
    FileNotFoundError if the model artifact does not exist — the caller
    (main.py) catches this and logs a warning rather than crashing the server,
    so the /health endpoint remains available even without a trained model.

    The Booster is cached separately because SHAP values are computed with
    booster.predict(pred_contribs=True) — XGBoost's built-in exact TreeSHAP.

    Side effects
    ------------
    Sets the module-level _model and _booster variables.

    Raises
    ------
    FileNotFoundError
        If models/xgb_rul.joblib is not present. Train the model by running
        notebooks/03_modeling.ipynb first.
    """
    global _model, _booster
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first by running notebooks/03_modeling.ipynb."
        )
    _model = joblib.load(MODEL_PATH)
    _booster = _model.get_booster()


def get_model() -> tuple:
    """
    Return the cached (model, booster) pair, loading them if necessary.

    Provides a lazy-load fallback for cases where load_model() was not called
    at startup (e.g. during testing). In production, load_model() is always
    called first via the lifespan handler, so this is effectively a no-op.

    Returns
    -------
    tuple[XGBRegressor, xgboost.Booster]
        The trained model and its underlying booster (used for SHAP).
    """
    if _model is None:
        load_model()
    return _model, _booster


def readings_to_dataframe(request: PredictRequest) -> pd.DataFrame:
    """
    Convert the API request payload into a feature-engineered DataFrame.

    Applies the same rolling mean and standard deviation transform used during
    training (src/features.add_rolling_features), computed over the full
    sequence of cycles in the request. The window is capped at the number of
    cycles provided so that short payloads still receive valid (if noisier)
    feature values.

    Parameters
    ----------
    request : PredictRequest
        Validated request payload containing an ordered list of SensorReading objects.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw sensor columns plus _mean30 and _std30 variants,
        ready for column selection and inference.
    """
    rows = [r.model_dump() for r in request.readings]
    df = pd.DataFrame(rows)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    window = min(30, len(df))

    rolled_mean = (
        df[sensor_cols]
        .rolling(window, min_periods=1)
        .mean()
        .add_suffix("_mean30")
    )
    rolled_std = (
        df[sensor_cols]
        .rolling(window, min_periods=1)
        .std()
        .fillna(0)
        .add_suffix("_std30")
    )
    return pd.concat([df, rolled_mean, rolled_std], axis=1)


def predict(request: PredictRequest) -> PredictResponse:
    """
    Run end-to-end inference for a single engine and return a structured response.

    The prediction is made from the most recent cycle (last row after feature
    engineering), because that row has the richest rolling-window context.
    SHAP values are computed for that single row; the top 5 by absolute
    magnitude are returned with their signed contributions in cycle units.

    Parameters
    ----------
    request : PredictRequest
        Validated request payload with at least one SensorReading.

    Returns
    -------
    PredictResponse
        Structured response including predicted RUL, a ±15-cycle confidence
        band, the top 5 SHAP attributions, the number of cycles provided,
        and an optional data-quality warning.

    Raises
    ------
    FileNotFoundError
        Propagated from load_model() if the model artifact is missing.
    Exception
        Any unexpected inference error is propagated to the caller (main.py),
        which maps it to an HTTP 500 response.
    """
    model, booster = get_model()

    df = readings_to_dataframe(request)

    # Exclude metadata and operating settings — must match training feature set.
    feature_cols = [
        c for c in df.columns
        if c not in {"cycle", "op_setting_1", "op_setting_2", "op_setting_3"}
    ]
    X = df[feature_cols].iloc[[-1]]  # Most recent cycle only

    raw_pred = float(model.predict(X)[0])
    predicted_rul = int(np.clip(raw_pred, 0, 125))

    # SHAP attribution for this single prediction — XGBoost's built-in exact
    # TreeSHAP. pred_contribs returns (1, n_features + 1); the final column is
    # the bias (expected value) term, so it is dropped before ranking.
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    contribs = booster.predict(dmat, pred_contribs=True)[0][:-1]
    raw_shap = pd.Series(contribs, index=feature_cols)
    top_5_features = raw_shap.abs().sort_values(ascending=False).head(5).index

    top_factors = [
        ShapEntry(
            feature=feat,
            value=round(float(raw_shap[feat]), 3),
            direction="increases_rul" if raw_shap[feat] > 0 else "decreases_rul",
        )
        for feat in top_5_features
    ]

    warning = None
    if len(request.readings) < 30:
        warning = (
            "Fewer than 30 cycles provided — rolling features may be unstable. "
            "Accuracy improves with more cycles."
        )

    # Cycle-order guard: readings are assumed earliest-to-latest. If the cycle
    # numbers are not non-decreasing, honour the request as given but say so.
    cycles = [r.cycle for r in request.readings]
    if any(b < a for a, b in zip(cycles, cycles[1:])):
        order_note = (
            "Cycle numbers are not in increasing order; readings were used as given."
        )
        warning = order_note if warning is None else f"{warning} {order_note}"

    return PredictResponse(
        predicted_rul=predicted_rul,
        confidence_band={"low": max(0, predicted_rul - 15), "high": min(125, predicted_rul + 15)},
        top_factors=top_factors,
        cycles_provided=len(request.readings),
        warning=warning,
    )
