"""
Pydantic models — request and response shapes for the prediction API.

Pydantic validates incoming JSON automatically. If a required field is
missing or the wrong type, FastAPI returns a 422 error before your code runs.
"""

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """One cycle's worth of sensor readings for a single engine."""
    cycle: int = Field(..., ge=1, description="Cycle number (1-indexed)")
    op_setting_1: float = Field(default=0.0)
    op_setting_2: float = Field(default=0.0)
    op_setting_3: float = Field(default=100.0)
    sensor_2: float = Field(..., description="LPC outlet temperature (°R)")
    sensor_3: float = Field(..., description="HPC outlet temperature (°R)")
    sensor_4: float = Field(..., description="LPT outlet temperature (°R)")
    sensor_7: float = Field(..., description="HPC outlet static pressure (psia)")
    sensor_8: float = Field(..., description="Fuel flow ratio (pps/psia)")
    sensor_9: float = Field(..., description="BPR")
    sensor_11: float = Field(..., description="HPC outlet coolant bleed (lbm/s)")
    sensor_12: float = Field(..., description="HPC outlet temperature (°R) — alt sensor")
    sensor_13: float = Field(..., description="HPT coolant bleed (lbm/s)")
    sensor_14: float = Field(..., description="LPT outlet coolant bleed (lbm/s)")
    sensor_15: float = Field(default=8.0)
    sensor_17: float = Field(default=390.0)
    sensor_20: float = Field(default=39.0)
    sensor_21: float = Field(default=23.0)


class PredictRequest(BaseModel):
    """
    A sequence of sensor readings for one engine.
    Minimum 30 cycles recommended for rolling features to stabilize.
    """
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        description="Ordered list of sensor readings, earliest to latest"
    )


class ShapEntry(BaseModel):
    feature: str
    value: float
    direction: str  # "increases_rul" | "decreases_rul"


class PredictResponse(BaseModel):
    predicted_rul: int = Field(..., description="Predicted remaining cycles before failure")
    confidence_band: dict = Field(..., description="Approximate low/high range")
    top_factors: list[ShapEntry] = Field(..., description="Top 5 SHAP-based explanations")
    cycles_provided: int
    warning: str | None = Field(default=None)
