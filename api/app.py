from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from chpp.data_sources.fred import load_hpi_series, resolve_series_id, summarize_changes
from chpp.data_sources.geo import default_payload_for_region, get_regions
from chpp.features import ENGINEERED_FEATURES
from chpp.predict import FEATURE_NAMES, predict_many, predict_one
from chpp.train import train_model

app = FastAPI(title="California Housing Price Predictor")

TEMPLATES_DIR = Path(__file__).parent / "templates"
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "models" / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "reports" / "metrics.json"


class HousingPayload(BaseModel):
    MedInc: float = Field(..., ge=0, description="Median income in tens of thousands")
    HouseAge: float = Field(..., ge=0, description="Median house age")
    AveRooms: float = Field(..., ge=0, description="Average rooms per household")
    AveBedrms: float = Field(..., ge=0, description="Average bedrooms per household")
    Population: float = Field(..., ge=0, description="Block population")
    AveOccup: float = Field(..., ge=0, description="Average household occupancy")
    Latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    Longitude: float = Field(..., ge=-180, le=180, description="Longitude")

    @field_validator("*")
    @classmethod
    def ensure_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Value must be finite.")
        return value


class PredictionResponse(BaseModel):
    prediction_hundreds_k: float
    prediction_usd: float
    rmse_hundreds_k: float | None
    rmse_usd: float | None
    units: str
    note: str


class BatchPredictionRequest(BaseModel):
    items: list[HousingPayload] = Field(..., min_length=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class RegionSummary(BaseModel):
    region_id: str
    name: str
    lat: float
    lon: float


class RegionComparisonItem(BaseModel):
    region: RegionSummary
    features: HousingPayload
    prediction: PredictionResponse


class RegionComparisonResponse(BaseModel):
    items: list[RegionComparisonItem]


class TrendPoint(BaseModel):
    date: str
    value: float


class TrendResponse(BaseModel):
    region_id: str
    series_id: str
    points: list[TrendPoint]
    latest_value: float | None
    change_12m_pct: float | None
    change_60m_pct: float | None


class Insight(BaseModel):
    title: str
    detail: str
    severity: str


class InsightsResponse(BaseModel):
    insights: list[Insight]
    trend: TrendResponse | None
    sensitivity: Dict[str, float]


@dataclass(frozen=True)
class ModelState:
    model: Any | None
    metrics: dict | None
    error: str | None
    trained_at: str | None
    source: str
    model_type: str | None


def _load_artifacts() -> tuple[Any, dict] | None:
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return model, metrics
    return None


def _metrics_rmse(metrics: dict | None) -> float | None:
    if not metrics:
        return None
    test = metrics.get("test", {})
    rmse = test.get("rmse")
    return float(rmse) if rmse is not None else None


@lru_cache
def get_model_state() -> ModelState:
    try:
        artifacts = _load_artifacts()
        if artifacts:
            model, metrics = artifacts
            return ModelState(
                model=model,
                metrics=metrics,
                error=None,
                trained_at=metrics.get("timestamp"),
                source="artifacts",
                model_type=metrics.get("model_name", type(model).__name__),
            )

        model, metrics = train_model()
        return ModelState(
            model=model,
            metrics=metrics,
            error=None,
            trained_at=metrics.get("timestamp"),
            source="trained-in-memory",
            model_type=metrics.get("model_name", type(model).__name__),
        )
    except Exception as exc:
        return ModelState(
            model=None,
            metrics=None,
            error=str(exc),
            trained_at=None,
            source="error",
            model_type=None,
        )


def _payload_to_dict(payload: HousingPayload) -> Dict[str, float]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def _build_prediction_response(y: float, rmse: float | None) -> PredictionResponse:
    return PredictionResponse(
        prediction_hundreds_k=y,
        prediction_usd=round(y * 100_000, 2),
        rmse_hundreds_k=float(rmse) if rmse is not None else None,
        rmse_usd=round(float(rmse) * 100_000, 2) if rmse is not None else None,
        units="1.0 = $100,000",
        note="RMSE computed on held-out test split (random_state=42)",
    )


def _region_summary(region) -> RegionSummary:
    return RegionSummary(
        region_id=region.region_id,
        name=region.name,
        lat=region.lat,
        lon=region.lon,
    )


def _trend_response(region_id: str, series_id: str, points_df) -> TrendResponse:
    points = [
        TrendPoint(date=row["date"].date().isoformat(), value=float(row["value"]))
        for _, row in points_df.iterrows()
    ]
    summary = summarize_changes(points_df)
    return TrendResponse(
        region_id=region_id,
        series_id=series_id,
        points=points,
        latest_value=summary.get("latest"),
        change_12m_pct=summary.get("change_12m_pct"),
        change_60m_pct=summary.get("change_60m_pct"),
    )


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((TEMPLATES_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
@app.get("/healthz")
def health() -> Dict[str, str]:
    state = get_model_state()
    if state.model is None:
        return {"status": "error", "detail": state.error or "model not ready"}
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> Dict:
    state = get_model_state()
    rmse = _metrics_rmse(state.metrics)
    return {
        "model_loaded": state.model is not None,
        "model_type": state.model_type,
        "training_data": "sklearn California Housing",
        "base_features": FEATURE_NAMES,
        "engineered_features": ENGINEERED_FEATURES,
        "rmse_hundreds_k": float(rmse) if rmse is not None else None,
        "rmse_usd": round(float(rmse) * 100_000, 2) if rmse is not None else None,
        "trained_at": state.trained_at,
        "source": state.source,
        "metrics": state.metrics,
        "error": state.error,
    }



@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HousingPayload) -> PredictionResponse:
    state = get_model_state()
    if state.model is None:
        raise HTTPException(status_code=500, detail=f"Model not ready: {state.error}")

    try:
        y = predict_one(state.model, _payload_to_dict(payload))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    rmse = _metrics_rmse(state.metrics)
    return _build_prediction_response(y, rmse)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    state = get_model_state()
    if state.model is None:
        raise HTTPException(status_code=500, detail=f"Model not ready: {state.error}")

    rmse = _metrics_rmse(state.metrics)
    try:
        payloads = [_payload_to_dict(item) for item in payload.items]
        preds = predict_many(state.model, payloads)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    responses = [_build_prediction_response(pred, rmse) for pred in preds]
    return BatchPredictionResponse(predictions=responses)


@app.get("/compare/locations", response_model=RegionComparisonResponse)
def compare_locations(region_ids: str = Query(default="")) -> RegionComparisonResponse:
    state = get_model_state()
    if state.model is None:
        raise HTTPException(status_code=500, detail=f"Model not ready: {state.error}")

    regions = get_regions()
    if region_ids:
        requested = {rid.strip() for rid in region_ids.split(",") if rid.strip()}
        regions = [region for region in regions if region.region_id in requested]
        if not regions:
            raise HTTPException(status_code=400, detail="No valid region_ids provided.")

    payloads = [default_payload_for_region(region.region_id) for region in regions]
    preds = predict_many(state.model, payloads)
    rmse = _metrics_rmse(state.metrics)

    items = []
    for region, pred, payload in zip(regions, preds, payloads):
        items.append(
            RegionComparisonItem(
                region=_region_summary(region),
                features=HousingPayload(**payload),
                prediction=_build_prediction_response(pred, rmse),
            )
        )
    return RegionComparisonResponse(items=items)


@app.get("/trends/hpi", response_model=TrendResponse)
def trends_hpi(region_id: str = Query(default="us"), max_points: int = Query(default=240, ge=24)):
    series = load_hpi_series(region_id=region_id)
    points_df = series.points.sort_values("date").tail(max_points)
    return _trend_response(region_id, series.series_id, points_df)


@app.post("/insights", response_model=InsightsResponse)
def insights(payload: HousingPayload, region_id: str | None = None) -> InsightsResponse:
    state = get_model_state()
    if state.model is None:
        raise HTTPException(status_code=500, detail=f"Model not ready: {state.error}")

    base_payload = _payload_to_dict(payload)
    base_pred = predict_one(state.model, base_payload)

    med_inc_up = dict(base_payload)
    med_inc_up["MedInc"] = med_inc_up["MedInc"] * 1.1
    med_inc_down = dict(base_payload)
    med_inc_down["MedInc"] = med_inc_down["MedInc"] * 0.9
    pred_up = predict_one(state.model, med_inc_up)
    pred_down = predict_one(state.model, med_inc_down)

    sensitivity = {
        "medinc_up_10pct_usd": round(pred_up * 100_000, 2),
        "medinc_down_10pct_usd": round(pred_down * 100_000, 2),
        "base_usd": round(base_pred * 100_000, 2),
    }

    insights = [
        Insight(
            title="Income sensitivity",
            detail="A 10% change in median income shifts predictions by "
            f"{round((pred_up - pred_down) * 100_000, 2)} USD.",
            severity="info",
        )
    ]

    trend = None
    if region_id:
        try:
            series_id = resolve_series_id(region_id)
            series = load_hpi_series(region_id=region_id)
            trend = _trend_response(region_id, series_id, series.points)
            if trend.change_12m_pct is not None:
                direction = "up" if trend.change_12m_pct > 0 else "down"
                insights.append(
                    Insight(
                        title="Market momentum",
                        detail=f"Local HPI is {direction} {round(abs(trend.change_12m_pct), 2)}% over 12 months.",
                        severity="warning" if abs(trend.change_12m_pct) > 10 else "info",
                    )
                )
        except Exception:
            insights.append(
                Insight(
                    title="Trend data unavailable",
                    detail="Unable to load HPI series; using model-only insights.",
                    severity="info",
                )
            )

    return InsightsResponse(insights=insights, trend=trend, sensitivity=sensitivity)


