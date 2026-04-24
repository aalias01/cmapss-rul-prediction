"""
Model loading and inference logic.

Separated from main.py so the FastAPI routes stay thin and readable.
This is the bridge between the trained model (from notebooks/src/) and the API.
"""

import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path

from api.schemas import PredictRequest, PredictResponse, ShapEntry

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_rul.joblib"

# Module-level cache — model loads once when the server starts, not on every request
_model = None
_explainer = None


def load_model():
    global _model, _explainer
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first by running notebooks/03_modeling.ipynb."
        )
    _model = joblib.load(MODEL_PATH)
    _explainer = shap.TreeExplainer(_model)


def get_model():
    if _model is None:
        load_model()
    return _model, _explainer


def readings_to_dataframe(request: PredictRequest) -> pd.DataFrame:
    """Convert API request payload into a feature-engineered DataFrame."""
    rows = [r.model_dump() for r in request.readings]
    df = pd.DataFrame(rows)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    window = min(30, len(df))

    rolled_mean = df[sensor_cols].rolling(window, min_periods=1).mean().add_suffix("_mean30")
    rolled_std = df[sensor_cols].rolling(window, min_periods=1).std().fillna(0).add_suffix("_std30")

    df = pd.concat([df, rolled_mean, rolled_std], axis=1)
    return df


def predict(request: PredictRequest) -> PredictResponse:
    model, explainer = get_model()

    df = readings_to_dataframe(request)

    # Use the last row — the most recent sensor reading is what we predict from
    feature_cols = [c for c in df.columns if c not in {"cycle", "op_setting_1", "op_setting_2", "op_setting_3"}]
    X = df[feature_cols].iloc[[-1]]

    raw_pred = float(model.predict(X)[0])
    predicted_rul = int(np.clip(raw_pred, 0, 125))

    # SHAP explanation for this single prediction
    shap_values = explainer.shap_values(X)[0]
    shap_series = pd.Series(shap_values, index=feature_cols).abs().sort_values(ascending=False)
    top_5 = shap_series.head(5)

    raw_shap = pd.Series(explainer.shap_values(X)[0], index=feature_cols)
    top_factors = [
        ShapEntry(
            feature=feat,
            value=round(float(raw_shap[feat]), 3),
            direction="increases_rul" if raw_shap[feat] > 0 else "decreases_rul",
        )
        for feat in top_5.index
    ]

    warning = None
    if len(request.readings) < 30:
        warning = "Fewer than 30 cycles provided — rolling features may be unstable. Accuracy improves with more cycles."

    return PredictResponse(
        predicted_rul=predicted_rul,
        confidence_band={"low": max(0, predicted_rul - 15), "high": min(125, predicted_rul + 15)},
        top_factors=top_factors,
        cycles_provided=len(request.readings),
        warning=warning,
    )
