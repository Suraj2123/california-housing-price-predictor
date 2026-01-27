from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
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
rmse = None
load_error = None

try:
    data = fetch_california_housing(as_frame=True)
    X = data.data[FEATURE_NAMES]
    y = data.target

    # KEEP the test set so we can compute RMSE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Compute RMSE on held-out test set
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

except Exception as e:
    load_error = str(e)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((TEMPLATES_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
@app.get("/healthz")
def health() -> Dict[str, str]:
    if model is None or rmse is None:
        return {"status": "error", "detail": load_error or "model not ready"}
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> Dict:
    if rmse is None:
        # Avoid crashing if something went wrong at startup
        return {
            "model_loaded": model is not None,
            "error": load_error or "rmse not available",
            "features": FEATURE_NAMES,
        }

    return {
        "model_loaded": model is not None,
        "model_type": "GradientBoostingRegressor",
        "training_data": "sklearn California Housing",
        "features": FEATURE_NAMES,
        "rmse_hundreds_k": float(rmse),
        "rmse_usd": round(float(rmse) * 100_000, 2),
        "trained_at_startup": True,
    }


@app.post("/predict")
def predict(payload: Dict[str, float]) -> Dict:
    if model is None or rmse is None:
        raise HTTPException(status_code=500, detail=f"Model not ready: {load_error}")

    try:
        x = np.array([[payload[f] for f in FEATURE_NAMES]], dtype=float)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e.args[0]}")

    try:
        y = float(model.predict(x)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "prediction_hundreds_k": y,
        "prediction_usd": round(y * 100_000, 2),
        "rmse_hundreds_k": float(rmse),
        "rmse_usd": round(float(rmse) * 100_000, 2),
        "units": "1.0 = $100,000",
        "note": "RMSE computed on held-out test split (random_state=42)",
    }


