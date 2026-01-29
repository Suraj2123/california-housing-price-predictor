from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

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


