# Model card: turbofan RUL predictor

The served artifact is `models/xgb_rul.joblib`, an XGBoost regressor that
estimates the remaining useful life (RUL) of a turbofan engine in cycles from
its sensor history. Every number below traces to this repo's README, notebooks,
or the FD001 dataset description.

## What it does, and for whom

Given an ordered sequence of per-cycle sensor readings for one engine, the model
returns a single RUL estimate in cycles, a heuristic range around it, and the
five features that moved the estimate most. It is built for a reliability or
predictive-maintenance audience who want to see a degradation model that reports
its own error against known answers. It is a demo and portfolio artifact, not a
system for making flight or maintenance decisions.

## Training data

NASA CMAPSS FD001. 100 engines run to failure, 20,631 training cycles, a single
operating condition and a single fault mode (high-pressure compressor
degradation). Labels use the standard piecewise-linear convention: RUL is capped
at 125 cycles, so early-life cycles are all treated as equally healthy rather
than letting the long healthy plateau dominate the loss.

## Features

The model uses the 14 informative sensor channels. Seven near-constant channels
(sensor_1, 5, 6, 10, 16, 18, 19) are dropped: variance analysis in `01_eda.ipynb`
showed they carry no signal in FD001, and keeping them would add dimensionality
without information. Each remaining sensor contributes three columns: its raw
value, its 30-cycle rolling mean, and its 30-cycle rolling standard deviation,
for 42 features total. The 30-cycle window was chosen to span roughly one HPC
fouling cycle. The prediction is made from the most recent cycle, the row with
the richest rolling-window context. Operating settings are carried in the payload
but are not model features.

## Evaluation

RMSE on the official FD001 test set, with all splits grouped by engine unit so no
engine appears in both train and test:

| Model | RMSE (FD001 test set) |
|---|---|
| XGBoost + rolling features (this model) | 15.85 |
| Zheng et al. 2017, deep LSTM | 16.14 |
| Ridge regression (this project's baseline) | 17.47 |
| Babu et al. 2016, CNN | 18.45 |

SHAP attributions were checked against the failure physics. The top feature,
`sensor_3_mean30` (30-cycle rolling mean of HPC outlet temperature), is the
expected thermodynamic signature of compressor fouling: as the compressor fouls,
efficiency drops, outlet temperature rises, and remaining life shortens.

## Limitations

- FD001 is simulated data, not field data.
- The 125-cycle cap means very healthy engines all read near "125", by design.
- The range shipped with each prediction is a fixed plus or minus 15 cycle
  heuristic. It is not a calibrated prediction interval.
- Single operating condition and single fault mode. FD002 through FD004 add
  operating regimes and fault modes this model has not seen, so its numbers do
  not transfer to them.

## Intended use

Demonstration and portfolio use. Not for operational, flight, or maintenance
decisions.
