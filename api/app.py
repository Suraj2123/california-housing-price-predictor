from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(title="California Housing Price Predictor")

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = Path(__file__).parent / "templates"

MODEL_PATH = ROOT / "artifacts" / "models" / "model.joblib"

# ------------------------------------------------------------------------------
# Load model at startup (fail fast if missing)
# ------------------------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    load_error = str(e)

# Feature order expected by the model (sklearn California Housing)
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(
        (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    )

# ------------------------------------------------------------------------------
# Health (Render needs this)
# ------------------------------------------------------------------------------
@app.get("/health")
@app.get("/healthz")
def health() -> Dict[str, str]:
    if model is None:
        return {"status": "error", "detail": "model not loaded"}
    return {"status": "ok"}

# ------------------------------------------------------------------------------
# Model info 
# ------------------------------------------------------------------------------
@app.get("/model-info")
def model_info() -> Dict:
    return {
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "features": FEATURE_NAMES,
        "framework": "scikit-learn",
        "notes": "End-to-end regression service with FastAPI",
    }

# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------
@app.post("/predict")
def predict(payload: Dict[str, float]) -> Dict:
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded: {load_error}",
        )

    # Validate payload
    try:
        x = np.array([[payload[f] for f in FEATURE_NAMES]], dtype=float)
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature: {e.args[0]}",
        )

    # Predict
    try:
        y = float(model.predict(x)[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        )

    return {
        "prediction": y,
        "units": "median_house_value",
    }
