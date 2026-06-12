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
    GET  /health    → Liveness check for Render keep-alive and uptime monitors
    POST /predict   → RUL prediction + SHAP explanation (see api/schemas.py)
    GET  /docs      → Auto-generated Swagger UI (FastAPI built-in)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictRequest, PredictResponse
from api import predictor


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
        "https://cmapss-rul-prediction.vercel.app",  # Production frontend
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
    the model — always responds even if the model failed to load at startup.
    """
    return {"status": "ok"}


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
    try:
        return predictor.predict(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
