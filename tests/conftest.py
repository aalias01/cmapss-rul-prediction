"""Shared pytest fixtures for the RUL serving-path tests.

The fixture engine is a 40-cycle slice of engine 1 from the official FD001
test set, cut to tests/fixtures/engine_fixture.csv so the suite runs offline
against the real committed artifact with no network and no Render.
"""

import csv
from pathlib import Path

import pytest

from api.schemas import PredictRequest

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "engine_fixture.csv"


def _load_readings(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {k: (int(float(v)) if k == "cycle" else float(v)) for k, v in row.items()}
        for row in rows
    ]


@pytest.fixture
def engine_readings() -> list[dict]:
    """Raw reading dicts for the fixture engine, earliest cycle first."""
    return _load_readings(FIXTURE_CSV)


@pytest.fixture
def predict_request(engine_readings: list[dict]) -> PredictRequest:
    """The fixture engine as a validated PredictRequest."""
    return PredictRequest(readings=engine_readings)
