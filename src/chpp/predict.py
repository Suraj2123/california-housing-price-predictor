from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("artifacts/models/model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first: python -m chpp.train"
        )
    return joblib.load(MODEL_PATH)


def predict_one(features: dict) -> float:
    model = load_model()
    X = pd.DataFrame([features])
    pred = model.predict(X)[0]
    return float(pred)

y = float(model.predict(x)[0])
return {
  "prediction_hundreds_k": y,
  "prediction_usd": round(y * 100_000, 2),
  "units": "California Housing target (1.0 = $100,000)"
}

