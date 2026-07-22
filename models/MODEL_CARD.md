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
for 42 features total. The 30-cycle window is a chosen smoothing horizon that
balances recent history against short-term noise. It is not a measured fouling
period or a validated optimum. The prediction is made from the most recent
cycle, the row with the richest rolling-window context. Operating settings are
carried in the payload but are not model features.

## Evaluation

RMSE on the official FD001 test set, with all splits grouped by engine unit so no
engine appears in both train and test.

### Direct local comparison

| Model | RMSE (FD001 test set) |
|---|---|
| XGBoost + rolling features (this model) | 15.85 |
| Ridge regression (this project's baseline) | 17.47 |

The Ridge result was produced in this repository under the same local data and
evaluation path, so it is the valid direct comparison.

### Literature context only

| Published model | Reported RMSE |
|---|---|
| Zheng et al. 2017, deep LSTM | 16.14 |
| Babu et al. 2016, CNN | 18.45 |

These published values provide historical context. This repository has not
audited protocol parity for preprocessing, RUL labeling, test aggregation, or
scoring, so neither literature value is a valid direct benchmark for 15.85.

The largest SHAP attribution is `sensor_3_mean30`, the 30-cycle rolling mean of
HPC outlet temperature. That association is consistent with a compressor
degradation interpretation, but SHAP explains this model's prediction. It does
not establish physical causality or independently confirm a fouling mechanism.

## Limitations

- FD001 is simulated data, not field data.
- The 125-cycle cap means very healthy engines all read near "125", by design.
- The range shipped with each prediction is a fixed plus or minus 15 cycle
  heuristic. It is not a calibrated prediction interval.
- The 30-cycle rolling window is a chosen smoothing horizon, not a validated
  physical fouling period.
- Literature RMSE values are unaudited context. Only the local Ridge RMSE of
  17.47 is a direct comparison under this repository's evaluation path.
- SHAP values are model attributions and associations, not causal evidence.
- Single operating condition and single fault mode. FD002 through FD004 add
  operating regimes and fault modes this model has not seen, so its numbers do
  not transfer to them.

## Intended use

Demonstration and portfolio use. Not for operational, flight, or maintenance
decisions.
