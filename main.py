"""FastAPI serving layer for daily M5 demand forecasts and price scenarios."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import date as Date
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from xgboost import XGBRegressor

from app.monitoring import PREDICTION_LATENCY, RECOMMENDATION_DELTA, REQUESTS, DriftMonitor
from pricing.features import FEATURE_COLUMNS, features_from_context, price_scenarios

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_PATH = MODEL_DIR / "pricing_model.json"
METADATA_PATH = MODEL_DIR / "metadata.json"


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="M5 Daily Dynamic Pricing API",
    description=(
        "Forecasts next-day sales from M5 daily history, then compares estimated "
        "revenue across candidate prices. This is an observational demo, not a causal "
        "or profit-optimizing pricing system."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
model: XGBRegressor | None = None
metadata: dict = {}
drift_monitor = DriftMonitor({})


class PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    item_id: str = Field(min_length=1, max_length=100)
    store_id: str = Field(min_length=1, max_length=20)
    date: Date = Field(description="Date for the next-day M5-style sales estimate")
    current_price: float = Field(gt=0, description="Current M5 sell_price")
    price_7d_ago: float = Field(gt=0)
    price_ma_28d: float = Field(gt=0, description="Mean sell_price over prior 28 days")
    sales_lag_1d: float = Field(ge=0)
    sales_lag_7d: float = Field(ge=0)
    sales_lag_28d: float = Field(ge=0)
    sales_ma_7d: float = Field(ge=0)
    sales_ma_28d: float = Field(ge=0)
    sales_std_28d: float = Field(ge=0)
    event_flag: float = Field(ge=0, le=1)
    snap_day: float = Field(ge=0, le=1)


class PricingResponse(BaseModel):
    item_id: str
    store_id: str
    date: str
    current_price: float
    recommended_price: float
    price_change_pct: float
    predicted_units_at_current_price: float
    predicted_units_at_recommended_price: float
    estimated_revenue_at_current_price: float
    estimated_revenue_at_recommended_price: float
    model_version: str
    drift_psi: dict[str, float]
    drift_alert: bool
    recommendation_note: str


def _load_model() -> None:
    global model, metadata, drift_monitor
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        model = None
        metadata = {}
        drift_monitor = DriftMonitor({})
        return
    loaded = XGBRegressor()
    loaded.load_model(MODEL_PATH)
    loaded_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if loaded_metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("Saved model feature schema does not match the serving code.")
    model = loaded
    metadata = loaded_metadata
    drift_monitor = DriftMonitor(loaded_metadata.get("reference_histograms", {}))


@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    route = request.url.path if request.url.path in {"/recommend", "/health", "/metrics"} else "other"
    REQUESTS.labels(route=route, status=str(response.status_code)).inc()
    return response


@app.get("/health")
def health() -> dict:
    ready = model is not None
    return {
        "status": "ready" if ready else "model_not_trained",
        "model_loaded": ready,
        "model_version": metadata.get("model_version"),
        "data_kind": metadata.get("report", {}).get("data_kind"),
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend", response_model=PricingResponse)
def recommend(payload: PricingRequest) -> PricingResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="M5 model is not available. Run python -m pricing.train --data-dir data/raw/m5 first.",
        )
    context = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    context["date"] = payload.date
    item_indices = metadata.get("item_indices", {})
    store_indices = metadata.get("store_indices", {})
    known_series = set(metadata.get("known_series", []))
    if f"{payload.item_id}::{payload.store_id}" not in known_series:
        raise HTTPException(
            status_code=422,
            detail="This item_id/store_id pair was not included in this model's M5 training series.",
        )
    try:
        feature_values = features_from_context(context, item_indices, store_indices)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    current_price = float(payload.current_price)
    candidates = np.linspace(current_price * 0.8, current_price * 1.2, 41)
    scenarios = price_scenarios(feature_values, candidates)
    with PREDICTION_LATENCY.time():
        predicted_units = np.maximum(model.predict(scenarios), 0.0)
    scenario_revenue = candidates * predicted_units
    best = int(np.argmax(scenario_revenue))
    current_feature = price_scenarios(feature_values, [current_price])
    current_units = max(float(model.predict(current_feature)[0]), 0.0)
    recommended_price = float(candidates[best])
    recommended_units = float(predicted_units[best])
    change_pct = (recommended_price - current_price) / current_price
    RECOMMENDATION_DELTA.observe(change_pct)
    psi = drift_monitor.observe(feature_values)
    return PricingResponse(
        item_id=payload.item_id,
        store_id=payload.store_id,
        date=payload.date.isoformat(),
        current_price=round(current_price, 2),
        recommended_price=round(recommended_price, 2),
        price_change_pct=round(change_pct, 4),
        predicted_units_at_current_price=round(current_units, 3),
        predicted_units_at_recommended_price=round(recommended_units, 3),
        estimated_revenue_at_current_price=round(current_price * current_units, 2),
        estimated_revenue_at_recommended_price=round(float(scenario_revenue[best]), 2),
        model_version=metadata.get("model_version", "unknown"),
        drift_psi=psi,
        drift_alert=any(score >= 0.20 for score in psi.values()),
        recommendation_note=(
            "Revenue scenario within ±20% of current price. M5 does not provide unit cost "
            "or competitor prices, and historical price associations are not causal."
        ),
    )
