from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

app = FastAPI(title="California Housing Price Predictor")

TEMPLATES_DIR = Path(__file__).parent / "templates"

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

model = None
load_error = None

try:
    data = fetch_california_housing(as_frame=True)
    X = data.data[FEATURE_NAMES]
    y = data.target

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
except Exception as e:
    load_error = str(e)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((TEMPLATES_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
@app.get("/healthz")
def health() -> Dict[str, str]:
    if model is None:
        return {"status": "error", "detail": load_error or "model not ready"}
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> Dict:
    return {
        "model_loaded": model is not None,
        "model_type": "GradientBoostingRegressor",
        "training_data": "sklearn California Housing",
        "features": FEATURE_NAMES,
        "trained_at_startup": True,
    }


@app.post("/predict")
def predict(payload: Dict[str, float]) -> Dict:
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not ready: {load_error}")

    try:
        x = np.array([[payload[f] for f in FEATURE_NAMES]], dtype=float)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e.args[0]}")

    try:
        y = float(model.predict(x)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"prediction": y, "units": "median_house_value"}
