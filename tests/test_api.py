"""
tests/test_api.py – Pytest tests for the FastAPI prediction service.

Requirements:
  - 3+ tests
  - At least one uses TestClient (FastAPI/Starlette test client)
  - Tests must NOT depend on the CSV files (we build small test data inline)
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# We mock the model load so tests don't require models/model.joblib
# ---------------------------------------------------------------------------

_mock_model = MagicMock()
_mock_model.predict.return_value = np.array([750.0])

# Patch joblib.load BEFORE importing app.main so _MODEL is set to mock
with patch("joblib.load", return_value=_mock_model):
    from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Valid request payload
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "nearby_shops": 80,
    "days_to_salary": 5,
    "footfall_index": 60,
    "distance_to_market_km": 3.5,
    "avg_ticket_rs": 2500,
    "other_atms_1km": 5,
    "atm_type": "mall",
    "area": "Market",
}


# ---------------------------------------------------------------------------
# Test 1: /health returns 200 and "status": "ok"
# ---------------------------------------------------------------------------

def test_health_endpoint():
    """GET /health should return 200 with status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body


# ---------------------------------------------------------------------------
# Test 2: /predict returns 200 with valid payload
# ---------------------------------------------------------------------------

def test_predict_valid_input():
    """POST /predict with valid payload should return 200 and a numeric prediction."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))


# ---------------------------------------------------------------------------
# Test 3: /predict rejects nearby_shops > 200 with HTTP 422
# ---------------------------------------------------------------------------

def test_predict_rejects_nearby_shops_above_200():
    """Client rule: nearby_shops > 200 must be rejected with 422."""
    bad_payload = {**VALID_PAYLOAD, "nearby_shops": 201}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 4: /predict rejects negative distance_to_market_km with HTTP 422
# ---------------------------------------------------------------------------

def test_predict_rejects_negative_distance():
    """Client rule: distance_to_market_km < 0 must be rejected with 422."""
    bad_payload = {**VALID_PAYLOAD, "distance_to_market_km": -1.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 5: /predict rejects invalid atm_type with HTTP 422
# ---------------------------------------------------------------------------

def test_predict_rejects_invalid_atm_type():
    """atm_type must be one of bank_branch, mall, standalone."""
    bad_payload = {**VALID_PAYLOAD, "atm_type": "kiosk"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 6: /predict rejects invalid area with HTTP 422
# ---------------------------------------------------------------------------

def test_predict_rejects_invalid_area():
    """area must be one of Market, Residential, Highway."""
    bad_payload = {**VALID_PAYLOAD, "area": "Rural"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 7: /predict rejects missing required field with HTTP 422
# ---------------------------------------------------------------------------

def test_predict_rejects_missing_field():
    """A payload missing a required field should return 422."""
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "atm_type"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 8: Cleaning pipeline – sentinel replacement
# ---------------------------------------------------------------------------

def test_clean_sentinel_values():
    """fix_sentinel_values should replace 9999 with the column median."""
    from src.atm_demand.clean import fix_sentinel_values

    df = pd.DataFrame({
        "distance_to_market_km": [1.0, 2.0, 9999.0, 3.0, 9999.0],
        "other": [1, 2, 3, 4, 5],
    })
    cleaned = fix_sentinel_values(df)
    assert 9999.0 not in cleaned["distance_to_market_km"].values
    # Median of [1.0, 2.0, 3.0] = 2.0 (after removing sentinels, pre-impute)
    assert cleaned["distance_to_market_km"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Test 9: Cleaning pipeline – target typo correction
# ---------------------------------------------------------------------------

def test_clean_target_typos():
    """fix_target_typos should divide values > 1200 by 10."""
    from src.atm_demand.clean import fix_target_typos

    df = pd.DataFrame({
        "withdrawal_thousand_rs": [600.0, 800.0, 7651.0, 9515.0, 500.0],
    })
    cleaned = fix_target_typos(df)
    assert cleaned["withdrawal_thousand_rs"].max() < 1200.0
    assert abs(cleaned.loc[2, "withdrawal_thousand_rs"] - 765.1) < 0.01


# ---------------------------------------------------------------------------
# Test 10: Cleaning pipeline – spelling variants
# ---------------------------------------------------------------------------

def test_clean_spelling_variants():
    """fix_spelling_variants should map all atm_type variants to 3 clean levels."""
    from src.atm_demand.clean import fix_spelling_variants

    df = pd.DataFrame({
        "atm_type": ["Mall", "STANDALONE", "bank_branch", "mal.", "sta.", "ban."],
        "area": ["Market", "Market", "Market", "Market", "Market", "Market"],
    })
    cleaned = fix_spelling_variants(df)
    assert set(cleaned["atm_type"].unique()) <= {"mall", "standalone", "bank_branch"}


# ---------------------------------------------------------------------------
# Test 11: PSI calculation
# ---------------------------------------------------------------------------

def test_psi_same_distribution():
    """PSI between identical distributions should be near 0."""
    from src.atm_demand.monitor import compute_psi

    rng = np.random.default_rng(42)
    data = rng.normal(50, 10, 1000)
    psi = compute_psi(data, data.copy())
    assert psi < 0.1


def test_psi_different_distribution():
    """PSI between very different distributions should be high."""
    from src.atm_demand.monitor import compute_psi

    rng = np.random.default_rng(42)
    ref = rng.normal(50, 10, 1000)
    cmp = rng.normal(100, 10, 1000)  # shifted by 50 units
    psi = compute_psi(ref, cmp)
    assert psi > 0.1
