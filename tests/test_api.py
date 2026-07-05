"""HTTP-level tests via FastAPI's TestClient against the real model artifact."""

from fastapi.testclient import TestClient

from api import predictor
from api.main import app

client = TestClient(app)


def test_health_reports_model_fields():
    predictor.load_model()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert isinstance(body["n_features"], int)


def test_predict_ok_on_fixture(engine_readings):
    r = client.post("/predict", json={"readings": engine_readings})
    assert r.status_code == 200
    body = r.json()
    assert "predicted_rul" in body
    assert body["cycles_provided"] == len(engine_readings)


def test_predict_missing_required_sensor_is_422():
    bad = {"readings": [{"cycle": 1, "sensor_3": 1589.7}]}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)


def test_predict_empty_readings_is_422():
    r = client.post("/predict", json={"readings": []})
    assert r.status_code == 422
