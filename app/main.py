"""
app/main.py – FastAPI prediction service for ATM Cash Demand (MP231).

Endpoints:
  GET  /health   → {"status": "ok", "model": "loaded"}
  POST /predict  → {"prediction": <float>}

Client rules (enforced via Pydantic Field validation → HTTP 422 on violation):
  - nearby_shops must be <= 200
  - distance_to_market_km must be >= 0

Logging: one JSON line per prediction written to stdout (never to a file).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# JSON logging setup (one line per prediction, to stdout)
# ---------------------------------------------------------------------------

# Standard LogRecord attributes (everything else passed via extra= is custom)
_STD_ATTRS = frozenset(vars(logging.LogRecord(
    name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
)).keys()) | {"message"}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Capture all custom attributes passed via extra={...}
        for key, value in vars(record).items():
            if key not in _STD_ATTRS and key not in payload:
                if isinstance(value, (int, float, str, bool)):
                    payload[key] = value
        return json.dumps(payload)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JsonFormatter())
logger = logging.getLogger("atm_demand_api")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Load the champion model once at startup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _REPO_ROOT / "models" / "model.joblib"

try:
    _MODEL = joblib.load(_MODEL_PATH)
    logger.info("Champion model loaded", extra={"model_path": str(_MODEL_PATH)})
except FileNotFoundError:
    logger.warning("Model file not found – /predict will fail until model is trained")
    _MODEL = None


# ---------------------------------------------------------------------------
# Pydantic request schema
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Features required for ATM withdrawal prediction.
    Client rules enforced here so invalid input returns HTTP 422 automatically.
    """
    nearby_shops: int = Field(
        ...,
        ge=0,
        le=200,
        description="Number of shops within 500 m. Must be 0–200.",
    )
    days_to_salary: int = Field(
        ...,
        ge=0,
        le=31,
        description="Days remaining until the 1st of the month.",
    )
    footfall_index: int = Field(
        ...,
        ge=0,
        le=100,
        description="Street footfall index (0–100).",
    )
    distance_to_market_km: float = Field(
        ...,
        ge=0.0,
        description="Distance to main market in km. Must be >= 0.",
    )
    avg_ticket_rs: int = Field(
        ...,
        ge=0,
        description="Average withdrawal amount last week (rupees).",
    )
    other_atms_1km: int = Field(
        ...,
        ge=0,
        description="Number of other ATMs within 1 km.",
    )
    atm_type: Literal["bank_branch", "mall", "standalone"] = Field(
        ...,
        description="ATM location type.",
    )
    area: Literal["Market", "Residential", "Highway"] = Field(
        ...,
        description="Area classification.",
    )


class PredictResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str
    model: str


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ATM Cash Demand API",
    description="Predicts next-day ATM withdrawals for Guwahati Cooperative Bank.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness and readiness probe endpoint."""
    model_status = "loaded" if _MODEL is not None else "not_loaded"
    return {"status": "ok", "model": model_status}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict daily ATM withdrawal for the given features.
    Returns prediction in thousand rupees.
    """
    # Build a single-row DataFrame matching training feature order
    row = pd.DataFrame([{
        "nearby_shops": req.nearby_shops,
        "days_to_salary": req.days_to_salary,
        "footfall_index": req.footfall_index,
        "distance_to_market_km": req.distance_to_market_km,
        "avg_ticket_rs": req.avg_ticket_rs,
        "other_atms_1km": req.other_atms_1km,
        "atm_type": req.atm_type,
        "area": req.area,
    }])

    prediction = float(_MODEL.predict(row)[0])

    # JSON log line (no personal customer data)
    logger.info(
        "prediction",
        extra={
            "event": "prediction",
            "nearby_shops": req.nearby_shops,
            "atm_type": req.atm_type,
            "area": req.area,
            "prediction_thousand_rs": round(prediction, 2),
        },
    )

    return {"prediction": prediction}
