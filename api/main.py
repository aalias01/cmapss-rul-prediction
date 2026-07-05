"""
FastAPI application — Turbofan Engine RUL Prediction API.

This file defines the application instance, CORS configuration, lifespan
model loading, and all HTTP routes. Route handlers are intentionally thin —
prediction logic lives in api/predictor.py, and request/response shapes are
defined in api/schemas.py.

Run locally
-----------
    uvicorn api.main:app --reload

Endpoints
---------
    GET  /          → API metadata (name, version, author, dataset)
    GET  /health    → Liveness check plus model_loaded / n_features status
    POST /predict   → RUL prediction + SHAP explanation (see api/schemas.py)
    GET  /docs      → Auto-generated Swagger UI (FastAPI built-in)
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import predictor
from api.schemas import PredictRequest, PredictResponse

# Log through uvicorn's error logger so per-request timing lines appear in the
# server output without a separate logging config.
logger = logging.getLogger("uvicorn.error")


def _log_predict(status_code: int, cycles_provided: int, started: float) -> None:
    """Log one line per /predict call: status, cycle count, latency in ms.

    Only the cycle count is recorded, never the payload contents.
    """
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "predict status=%s cycles_provided=%s ms=%s",
        status_code, cycles_provided, elapsed_ms,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — load the model once at server startup.

    Loading the XGBoost artifact and caching its booster (for TreeSHAP) here
    ensures the first prediction request is fast. If the model file is not
    found (e.g. the artifact has not been trained yet), a warning is logged
    but the server continues running so /health remains reachable.
    """
    try:
        predictor.load_model()
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    yield


app = FastAPI(
    title="Turbofan Engine RUL Predictor",
    description=(
        "Predicts Remaining Useful Life (RUL) of turbofan engines from sensor readings. "
        "Built on the NASA CMAPSS FD001 dataset using XGBoost with SHAP interpretability. "
        "By Alvin Alias — MS Data Science, University of Washington."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows the Vercel frontend to call this API from the browser.
# Update allow_origins with your actual Vercel URL after deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://turbofan.alvinalias.com",           # canonical demo (Primary)
        "https://cmapss-rul-prediction.vercel.app",  # legacy, 308-redirects to subdomain
        "http://localhost:3000",             # Local frontend dev server
        "http://localhost:5173",             # Vite/static preview fallback
        "http://localhost:5500",             # Local static server
        "http://127.0.0.1:5500",            # VS Code Live Server
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    """Return API metadata. Useful for quick sanity checks after deployment."""
    return {
        "name": "Turbofan RUL Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "predict": "/predict",
        "author": "Alvin Alias",
        "dataset": "NASA CMAPSS FD001",
    }


@app.get("/health")
def health() -> dict:
    """
    Liveness check endpoint.

    Used by Render's health check system to confirm the service is running,
    and by external uptime monitors. Returns immediately without touching
    the model on the request path — it only reads the cached module globals,
    so it always responds even if the model failed to load at startup.

    Fields
    ------
    status : str
        Always "ok" while the server is up (backward compatible).
    model_loaded : bool
        Whether the XGBoost artifact is loaded and ready to serve predictions.
    n_features : int | None
        Number of features the booster expects, or null when no model is loaded.
    """
    model_loaded = predictor._model is not None
    n_features = (
        len(predictor._booster.feature_names)
        if model_loaded and predictor._booster is not None
        else None
    )
    return {"status": "ok", "model_loaded": model_loaded, "n_features": n_features}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict remaining useful life from a sequence of turbofan sensor readings.

    Accepts an ordered list of per-cycle sensor readings for one engine,
    applies rolling feature engineering, runs XGBoost inference on the most
    recent cycle, and returns the predicted RUL with a SHAP-based explanation.

    Provide at least 30 cycles of readings for rolling features to stabilise.
    Readings must be ordered from the earliest cycle to the most recent.

    Returns a 503 if the model artifact has not been loaded (not trained yet),
    or a 500 for any unexpected inference error.
    """
    started = time.perf_counter()
    try:
        response = predictor.predict(request)
    except FileNotFoundError as e:
        _log_predict(503, len(request.readings), started)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _log_predict(500, len(request.readings), started)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    _log_predict(200, response.cycles_provided, started)
    return response
