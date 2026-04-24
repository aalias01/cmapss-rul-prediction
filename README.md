# Turbofan Engine RUL Prediction — NASA CMAPSS

**Domain:** Aerospace / Industrial Predictive Maintenance  
**Type:** Regression — Remaining Useful Life (RUL)  
**Dataset:** [NASA CMAPSS FD001](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)  
**Stack:** Python · XGBoost · SHAP · FastAPI · Vercel · Render

🔗 **[Live Demo](https://your-project.vercel.app)** &nbsp;|&nbsp; 📡 **[API Docs](https://your-api.onrender.com/docs)**

---

## Problem Statement

Turbofan engines degrade over time due to compressor fouling, thermal stress, and mechanical wear. Unplanned failures are catastrophic — operationally and financially. Given a sequence of sensor readings from an operating engine, **how many cycles remain before failure?**

This is the core problem of Remaining Useful Life prediction, a fundamental challenge in predictive maintenance across aerospace, manufacturing, and energy.

---

## Architecture

```
frontend/          →  Vercel (static site, no build step)
api/               →  Render (FastAPI Python backend)
src/               →  Shared feature engineering + model logic
notebooks/         →  EDA, feature engineering, training (exploratory)
```

```
User uploads CSV
      ↓
frontend/app.js  →  POST /predict  →  api/predictor.py
                                            ↓
                                    Feature engineering
                                    XGBoost inference
                                    SHAP explanation
                                            ↓
                              JSON response → rendered in browser
```

---

## Dataset

- **Source:** NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
- **Subset:** FD001 — single operating condition, single fault mode (HPC degradation)
- **Size:** 100 engines × run-to-failure, 21 sensors per cycle
- **RUL cap:** 125 cycles (standard piecewise-linear assumption — engines beyond 125 cycles have effectively the same long-term RUL operationally)

---

## Approach

| Step | What | Why |
|------|------|-----|
| Sensor filtering | Drop 7 near-constant sensors | Zero variance = zero signal |
| Feature engineering | Rolling mean + std over 30-cycle window per engine | Smooth noise; capture trend |
| Train/test split | Grouped by engine unit | Prevent temporal leakage |
| Baseline | Linear regression | Benchmark reference point |
| Primary model | XGBoost regressor | Industry-standard for tabular; fast; interpretable |
| Interpretability | SHAP TreeExplainer | Per-prediction feature attribution |
| API | FastAPI on Render | Clean JSON interface; auto-generates /docs |
| Frontend | Vanilla HTML/CSS/JS on Vercel | Professional UI; no framework overhead |

---

## Results

| Model | RMSE | Notes |
|-------|------|-------|
| Linear Regression | — | Baseline (TBD) |
| XGBoost | — | Primary model (TBD) |
| Published benchmark | ~18.0 | BiLSTM, Zheng et al. 2017 |

---

## Running Locally

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate cmapss-rul
```

### 2. Download the data

Place these files in `data/raw/`:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

Download: [NASA PCOE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) or [Kaggle mirror](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

### 3. Run notebooks in order

```bash
jupyter notebook
# 01_eda.ipynb → 02_feature_engineering.ipynb → 03_modeling.ipynb
```

### 4. Start the API

```bash
uvicorn api.main:app --reload
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 5. Open the frontend

Open `frontend/index.html` in a browser (or use VS Code Live Server).  
Update `API_BASE` in `frontend/app.js` to `http://localhost:8000` for local testing.

---

## Deployment

**Backend (Render):**
1. Push repo to GitHub
2. New Web Service on Render → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

**Frontend (Vercel):**
1. Import GitHub repo on Vercel
2. Set root directory to `frontend/`
3. No build step — deploy as static files
4. Update `API_BASE` in `app.js` to your Render URL before pushing

---

## Project Structure

```
.
├── environment.yml          ← Local dev (conda)
├── requirements.txt         ← Deployment (Render/pip)
├── .gitignore
│
├── data/raw/                ← gitignored — download separately
├── notebooks/               ← Exploratory analysis and training
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/                     ← Reusable Python modules
│   ├── features.py          ← Feature engineering pipeline
│   ├── model.py             ← Training + evaluation
│   └── utils.py             ← Plotting helpers
│
├── api/                     ← FastAPI backend → Render
│   ├── main.py              ← Routes and CORS
│   ├── schemas.py           ← Pydantic request/response models
│   └── predictor.py         ← Model loading + inference + SHAP
│
├── frontend/                ← Static site → Vercel
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── models/                  ← gitignored — trained model artifacts
```

---

## Skills Demonstrated

- Time-series feature engineering from sensor data (rolling statistics, degradation indicators)
- Temporal train/test splitting — grouped by engine unit to prevent leakage
- Gradient boosting regression (XGBoost) with hyperparameter tuning
- Model interpretability with SHAP TreeExplainer
- REST API design with FastAPI and Pydantic validation
- Professional frontend deployment on Vercel (no framework)
- Python backend deployment on Render with CORS configuration
- Clean reproducible environment with conda + `environment.yml`

---

*Built by Alvin Alias | MS Data Science, University of Washington | 2026*
