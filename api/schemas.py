"""
Pydantic request and response models for the RUL prediction API.

Pydantic validates all incoming JSON automatically. If a required field is
missing, the wrong type, or out of range, FastAPI returns a 422 Unprocessable
Entity response before any prediction code runs — no manual validation needed.

Model hierarchy
---------------
PredictRequest
    └── readings: list[SensorReading]   (one entry per cycle, ordered earliest → latest)

PredictResponse
    ├── predicted_rul: int
    ├── confidence_band: dict            (low / high range, ±15 cycles)
    ├── top_factors: list[ShapEntry]     (top 5 SHAP attributions)
    ├── cycles_provided: int
    └── warning: str | None             (set when < 30 cycles provided)
"""

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """
    One cycle's worth of sensor readings for a single engine.

    Only the 14 informative sensor channels are required. The 7 near-constant
    sensors (sensor_1, 5, 6, 10, 16, 18, 19) identified in EDA are omitted —
    they are not present in the trained model's feature matrix.

    Sensor names and physical interpretations follow Saxena & Goebel (2008).
    Operating settings default to the FD001 single-condition values and are
    not used as model features, but are retained in the payload for completeness.
    """
    cycle: int = Field(..., ge=1, description="Cycle number (1-indexed, earliest to latest)")
    op_setting_1: float = Field(default=0.0,   description="Operating setting 1")
    op_setting_2: float = Field(default=0.0,   description="Operating setting 2")
    op_setting_3: float = Field(default=100.0, description="Operating setting 3 (always 100 in FD001)")
    sensor_2:  float = Field(..., description="LPC outlet temperature (°R)")
    sensor_3:  float = Field(..., description="HPC outlet temperature (°R)")
    sensor_4:  float = Field(..., description="LPT outlet temperature (°R)")
    sensor_7:  float = Field(..., description="HPC outlet static pressure (psia)")
    sensor_8:  float = Field(..., description="Fuel flow ratio (pps/psia)")
    sensor_9:  float = Field(..., description="Bypass ratio (BPR)")
    sensor_11: float = Field(..., description="HPC outlet coolant bleed (lbm/s)")
    sensor_12: float = Field(..., description="HPC outlet temperature — alternate sensor (°R)")
    sensor_13: float = Field(..., description="HPT coolant bleed (lbm/s)")
    sensor_14: float = Field(..., description="LPT outlet coolant bleed (lbm/s)")
    sensor_15: float = Field(default=8.0,   description="Fan inlet static pressure (psia)")
    sensor_17: float = Field(default=390.0, description="HPT coolant bleed (lbm/s) — alternate")
    sensor_20: float = Field(default=39.0,  description="Bypass ratio — alternate")
    sensor_21: float = Field(default=23.0,  description="Demand fan speed")


class PredictRequest(BaseModel):
    """
    A time-ordered sequence of sensor readings for one engine.

    Provide at least 30 cycles for the rolling features to stabilise.
    Fewer cycles are accepted but will trigger a warning in the response.
    Readings must be ordered from earliest cycle to latest — the model
    always predicts from the final (most recent) reading.
    """
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        description="Ordered list of sensor readings, earliest cycle first",
    )


class ShapEntry(BaseModel):
    """
    A single SHAP feature attribution for the current prediction.

    The value is the SHAP contribution in RUL-cycle units — positive means
    the feature pushed the prediction toward a longer remaining life,
    negative means it pushed toward a shorter one.
    """
    feature:   str   = Field(..., description="Feature name (e.g. 'sensor_2_mean30')")
    value:     float = Field(..., description="SHAP value in cycle units (signed)")
    direction: str   = Field(..., description="'increases_rul' or 'decreases_rul'")


class PredictResponse(BaseModel):
    """
    Prediction response returned by POST /predict.

    predicted_rul is always an integer in [0, 125] (clipped to the label range).
    confidence_band is a ±15-cycle heuristic band — not a formal prediction interval.
    top_factors contains the five features with the largest absolute SHAP values
    for this specific prediction, along with their signed contributions.
    """
    predicted_rul:    int              = Field(..., description="Predicted remaining cycles before failure (0–125)")
    confidence_band:  dict             = Field(..., description="Approximate range: {'low': int, 'high': int}")
    top_factors:      list[ShapEntry]  = Field(..., description="Top 5 SHAP feature attributions for this prediction")
    cycles_provided:  int              = Field(..., description="Number of cycles in the input payload")
    warning:          str | None       = Field(default=None, description="Non-null when prediction quality may be reduced")
