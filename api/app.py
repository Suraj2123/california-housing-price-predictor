from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from chpp.predict import predict_one

app = FastAPI(title="California Housing Price Predictor", version="0.1.0")


class PredictRequest(BaseModel):
    MedInc: float = Field(..., ge=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., ge=0)
    AveBedrms: float = Field(..., ge=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., ge=0)
    Latitude: float
    Longitude: float


class PredictResponse(BaseModel):
    prediction: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred = predict_one(req.model_dump())
    return PredictResponse(prediction=pred)
