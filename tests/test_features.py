"""Tests for readings_to_dataframe, the request-side feature transform.

These guard the properties the model depends on: every sensor gets its
rolling mean/std pair, constant columns produce 0 rather than NaN, the
window never exceeds the payload length, and no rows are dropped.
"""

from api.predictor import readings_to_dataframe
from api.schemas import PredictRequest

ROLL_SUFFIXES = ("_mean30", "_std30")


def _raw_sensor_cols(df) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith("sensor_") and not c.endswith(ROLL_SUFFIXES)
    ]


def test_rolling_columns_created_for_every_sensor(predict_request):
    df = readings_to_dataframe(predict_request)
    for col in _raw_sensor_cols(df):
        assert f"{col}_mean30" in df.columns
        assert f"{col}_std30" in df.columns


def test_row_count_preserved(predict_request):
    df = readings_to_dataframe(predict_request)
    assert len(df) == len(predict_request.readings)


def test_std_of_constant_column_is_zero_not_nan(engine_readings):
    # Repeat one reading so every sensor column is constant; std must be 0.
    reading = engine_readings[0]
    req = PredictRequest(readings=[reading, reading, reading])
    df = readings_to_dataframe(req)
    std_cols = [c for c in df.columns if c.endswith("_std30")]
    assert std_cols
    assert df[std_cols].isna().to_numpy().sum() == 0
    assert (df[std_cols].iloc[-1] == 0).all()


def test_window_caps_at_payload_length(engine_readings):
    # A payload shorter than the 30-cycle window still yields features and rows.
    req = PredictRequest(readings=engine_readings[:5])
    df = readings_to_dataframe(req)
    assert len(df) == 5
    assert any(c.endswith("_mean30") for c in df.columns)
