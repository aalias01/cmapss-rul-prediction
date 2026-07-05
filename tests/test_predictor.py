"""Tests for predict(), the end-to-end serving path.

The feature-count test is the important one: it would catch a training and
serving feature mismatch, the classic silent RUL bug, before a visitor ever
runs the demo.
"""

from api import predictor
from api.schemas import PredictRequest


def test_predicted_rul_in_range(predict_request):
    resp = predictor.predict(predict_request)
    assert 0 <= resp.predicted_rul <= 125


def test_confidence_band_clamped_and_consistent(predict_request):
    resp = predictor.predict(predict_request)
    low = resp.confidence_band["low"]
    high = resp.confidence_band["high"]
    assert 0 <= low <= 125
    assert 0 <= high <= 125
    assert low <= resp.predicted_rul <= high
    assert low == max(0, resp.predicted_rul - 15)
    assert high == min(125, resp.predicted_rul + 15)


def test_exactly_five_top_factors_with_matching_direction(predict_request):
    resp = predictor.predict(predict_request)
    assert len(resp.top_factors) == 5
    for factor in resp.top_factors:
        expected = "increases_rul" if factor.value > 0 else "decreases_rul"
        assert factor.direction == expected


def test_cycles_provided_matches_input(predict_request, engine_readings):
    resp = predictor.predict(predict_request)
    assert resp.cycles_provided == len(engine_readings)


def test_feature_count_matches_booster(predict_request):
    _, booster = predictor.get_model()
    df = predictor.readings_to_dataframe(predict_request)
    feature_cols = [
        c for c in df.columns
        if c not in {"cycle", "op_setting_1", "op_setting_2", "op_setting_3"}
    ]
    assert len(feature_cols) == len(booster.feature_names)


def test_short_request_sets_warning(engine_readings):
    req = PredictRequest(readings=engine_readings[:10])
    resp = predictor.predict(req)
    assert resp.warning is not None
    assert "30 cycles" in resp.warning


def test_ordered_request_has_no_order_warning(predict_request):
    resp = predictor.predict(predict_request)
    if resp.warning:
        assert "not in increasing order" not in resp.warning


def test_out_of_order_request_appends_order_warning(engine_readings):
    shuffled = list(engine_readings)
    shuffled[0], shuffled[-1] = shuffled[-1], shuffled[0]
    resp = predictor.predict(PredictRequest(readings=shuffled))
    assert resp.warning is not None
    assert "not in increasing order" in resp.warning
