"""
FastAPI application — Turbofan RUL Prediction API

Run locally:
    uvicorn api.main:app --reload

Endpoints:
    GET  /          → API info
    GET  /health    → Health check (Render keep-alive)
    POST /predict   → RUL prediction + SHAP explanation
    GET  /docs      → Auto-generated interactive API docs (Swagger UI)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictRequest, PredictResponse
from api import predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup so first request is fast."""
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

# CORS — allows the Vercel frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-project.vercel.app",  # Update with your Vercel URL
        "http://localhost:3000",             # Local frontend dev
        "http://127.0.0.1:5500",            # VS Code Live Server
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "Turbofan RUL Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "predict": "/predict",
        "author": "Alvin Alias",
        "dataset": "NASA CMAPSS FD001",
    }


@app.get("/health")
def health():
    """Keep-alive endpoint for Render free tier. Also used for uptime monitoring."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict remaining useful life from a sequence of sensor readings.

    - Provide at least 30 cycles of readings for best accuracy
    - Readings must be ordered earliest to latest
    - Returns predicted RUL, a confidence band, and top 5 SHAP explanations
    """
    try:
        return predictor.predict(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
