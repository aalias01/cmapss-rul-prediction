# Turbofan Engine RUL Prediction (NASA CMAPSS)

Predicts how many cycles a turbofan engine has left before failure, from raw sensor readings. XGBoost with rolling-window features reaches 15.85 RMSE on the official FD001 test set, ahead of the widely cited Zheng et al. 2017 deep LSTM (16.14), using tabular features only. Per-prediction SHAP output maps to known HPC degradation physics.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-FF6600)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-exact_TreeSHAP-7B2FBE)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

**[Live demo](https://turbofan.alvinalias.com)** | **[API docs](https://cmapss-rul-api.onrender.com/docs)**

## Results

| Model | RMSE (FD001 test set) | Reference |
|-------|----------------------|-----------|
| Babu et al. 2016, CNN | 18.45 | First CNN for RUL on CMAPSS |
| Ridge regression (baseline) | 17.47 | This project's linear baseline |
| Zheng et al. 2017, deep LSTM | 16.14 | Widely cited sequence-model benchmark |
| XGBoost + rolling features | **15.85** | This project |

Transformer and hybrid CNN-LSTM papers published since have pushed FD001 lower, so 15.85 is a strong classical-ML result, not current state of the art.

Dataset: 100 engines run to failure, 20,631 training cycles, 14 informative sensors after dropping 7 near-zero-variance channels, single operating condition (FD001), RUL capped at 125 cycles per the standard piecewise-linear convention.

<p align="center">
  <img src="figures/02_predicted_vs_actual.png" width="420" alt="Predicted vs actual RUL scatter, XGBoost FD001 test set">
  &nbsp;&nbsp;
  <img src="figures/04_shap_bar.png" width="420" alt="Top 10 SHAP features by mean absolute value">
</p>

## Design decisions

**Split by engine, never randomly.** A random 80/20 split puts cycles from the same engine in both train and test, so the model memorizes engine-level drift instead of learning degradation. All splits here are grouped by engine unit.

**Features come from the physics, not grid search.** The 7 dropped sensors were identified by variance analysis and cross-checked against the CMAPSS dataset description (Saxena & Goebel 2008). The 30-cycle rolling window was chosen to span roughly one HPC fouling cycle.

**SHAP output is checked against known failure physics.** The top feature, `sensor_3_mean30` (30-cycle rolling mean of HPC outlet temperature), is the primary thermodynamic signature of compressor fouling in FD001: as the compressor fouls, efficiency drops, outlet temperature rises, and remaining life shortens. The next features (`sensor_2_mean30`, `sensor_11`, `sensor_9_mean30`, `sensor_14_mean30`) are the correlated upstream and downstream sensors responding to the same fault. The model learned that chain from data; SHAP makes it legible.

<p align="center">
  <img src="figures/01_degradation_trajectories.png" width="820" alt="Sensor degradation trajectories for 20 FD001 engines">
</p>

*sensor_3 (HPC outlet temperature) and sensor_2 (LPC outlet temperature) over full engine lifetimes for 20 training engines. The drift is noisy cycle to cycle but clear over 30-cycle windows, which is what the rolling mean captures.*

<p align="center">
  <img src="figures/03_shap_summary.png" width="700" alt="SHAP beeswarm summary, top 15 features">
</p>

*SHAP beeswarm for the top 15 features across 300 training samples. For `sensor_3_mean30`, high temperature (red) pushes predictions toward shorter RUL, the expected fouling signature.*

## How it works

```
User uploads CSV
      |
frontend/app.js  ->  POST /predict  ->  api/predictor.py
                                          feature engineering
                                          XGBoost inference
                                          SHAP explanation
                                              |
                                    JSON response rendered in browser
```

The API validates inputs with Pydantic before any inference runs and returns a confidence band with each point estimate. SHAP uses XGBoost's native `pred_contribs` (exact TreeSHAP), which also sidesteps the shap package's XGBoost 3.x loader incompatibility. Frontend is static HTML/CSS/JS on Vercel; backend is FastAPI on Render with a `render.yaml` Blueprint.

## Tech stack

Python 3.11, Pandas, NumPy, scikit-learn (Ridge baseline, grouped splits), XGBoost 2.0+, exact TreeSHAP via `pred_contribs`, FastAPI + Pydantic on Render, vanilla HTML/CSS/JS on Vercel. Pins in `environment.yml` (conda, local) and `requirements.txt` (pip, Render).

## Run it locally

```bash
conda env create -f environment.yml
conda activate cmapss-rul
```

Download FD001 from [NASA PCOE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) or the [Kaggle mirror](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place in `data/raw/` (gitignored):

```
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

Run the notebooks in order:

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Sensor distributions, degradation trajectories, variance analysis |
| `02_feature_engineering.ipynb` | Rolling features, train/test alignment, export processed data |
| `03_modeling.ipynb` | Grouped split, Ridge baseline, XGBoost, SHAP, save `models/xgb_rul.joblib` |

Start the API and frontend:

```bash
uvicorn api.main:app --reload
# docs at http://localhost:8000/docs
```

Open `frontend/index.html` in a browser and set `API_BASE` in `frontend/app.js` to `http://localhost:8000`. `frontend/sample_engine.csv` lets you try the upload flow without your own data.

## Limitations

- FD001 only: single operating condition, single fault mode (HPC degradation). FD002 through FD004 add operating regimes and fault modes this model hasn't seen.
- The RUL cap at 125 cycles means very healthy engines all read as "125+", by design.
- The API runs on Render's free tier; the first request after idle can take around a minute to cold-start.

## Deployment

Render: **New + > Blueprint**, connect the repo; `render.yaml` provisions build, start, health check, and the Python 3.11.9 pin. Manual fallback: build `pip install -r requirements.txt`, start `uvicorn api.main:app --host 0.0.0.0 --port $PORT`, health check `/health`. Vercel: import the repo, root directory `frontend/`, no build step, then set `API_BASE` in `app.js` and add the Vercel URL to `allow_origins` in `api/main.py`.

## Project structure

```
├── notebooks/       # 01 EDA, 02 features, 03 modeling
├── src/             # features.py, model.py, utils.py
├── api/             # FastAPI: main.py, schemas.py, predictor.py
├── frontend/        # static site + sample_engine.csv
├── figures/         # plots embedded above
├── data/raw/        # gitignored; download from NASA PCOE
└── models/          # xgb_rul.joblib (served by the API)
```

## Dataset and credits

- Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set.* NASA Ames Prognostics Data Repository. [link](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- Saxena, A., Goebel, K., Larrosa, C., & Luo, J. (2010). *Metrics for Evaluating Performance of Prognostic Techniques.* PHM 2010.
- Babu, G. S., Zhao, P., & Li, X.-L. (2016). *Deep CNN Based Regression Approach for Estimation of RUL.* DASFAA 2016.
- Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). *Long Short-Term Memory Network for Remaining Useful Life Estimation.* ICPHM 2017.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016. [arXiv](https://arxiv.org/abs/1603.02754)
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. [arXiv](https://arxiv.org/abs/1705.07874)
- Lundberg, S. M., et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees.* Nature Machine Intelligence. [arXiv](https://arxiv.org/abs/1905.04610)

NASA CMAPSS data is publicly available from the NASA Prognostics Data Repository and keeps its own terms; the MIT license here covers this repo's code only.

Built by Alvin Alias, MS Data Science, University of Washington.
