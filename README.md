# California Housing Price Predictor

Production-style machine learning regression service demonstrating end-to-end model training, evaluation, and deployment with a live interactive demo.

This project treats evaluation and uncertainty as first-class concerns, returning both predictions and a held-out test RMSE to communicate typical error — mirroring real-world ML engineering practices.

---

## Live Demo

- Web App: https://california-housing-price-predictor-m8ks.onrender.com/
- API Docs (Swagger): https://california-housing-price-predictor-m8ks.onrender.com/docs
- Model Info: https://california-housing-price-predictor-m8ks.onrender.com/model-info

### Example Prediction

    curl -X POST "https://california-housing-price-predictor-m8ks.onrender.com/predict" \
      -H "Content-Type: application/json" \
      -d '{"MedInc":5,"HouseAge":25,"AveRooms":5.5,"AveBedrms":1,"Population":1200,"AveOccup":2.8,"Latitude":34.05,"Longitude":-118.25}'

Example response:

    {
      "prediction_hundreds_k": 2.4383133660989826,
      "prediction_usd": 243831.34,
      "rmse_hundreds_k": 0.5422152016168362,
      "rmse_usd": 54221.52,
      "units": "1.0 = $100,000",
      "note": "RMSE computed on held-out test split (random_state=42)"
    }

Interpreting the output:
- prediction_usd is the predicted median house value in USD
- rmse_usd represents typical error based on a held-out test set

---

## Problem Overview

Predict the median house value for California census tracts using demographic and geographic features.

This is framed as a regression problem where:
- the output is continuous
- predictions should never be shown without uncertainty context
- model quality must be quantified and exposed

---

## Dataset

- Source: sklearn.datasets.fetch_california_housing
- Target: Median house value (scaled so that 1.0 = $100,000)
- Features:
  - MedInc — Median income
  - HouseAge — Median house age
  - AveRooms — Average rooms per household
  - AveBedrms — Average bedrooms per household
  - Population — Block population
  - AveOccup — Average household occupancy
  - Latitude
  - Longitude

---

## Model and Evaluation

- Model: GradientBoostingRegressor
- Framework: scikit-learn
- Train/Test Split: 80 / 20
- Random Seed: 42
- Metric: RMSE (Root Mean Squared Error)

RMSE is computed on a held-out test split and exposed via the API to communicate typical prediction error. This reflects a production ML mindset: predictions without uncertainty are incomplete.

---

## API

### Endpoints

- GET /            — Interactive web UI
- GET /docs        — Swagger API documentation
- GET /health      — Health check
- GET /healthz     — Health check (Render)
- GET /model-info  — Model metadata and evaluation metrics
- POST /predict    — Predict median house value

### Request body for POST /predict

    {
      "MedInc": 5,
      "HouseAge": 25,
      "AveRooms": 5.5,
      "AveBedrms": 1,
      "Population": 1200,
      "AveOccup": 2.8,
      "Latitude": 34.05,
      "Longitude": -118.25
    }

---

## Local Development

1. Create and activate a virtual environment

    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip

2. Install dependencies

    pip install -r requirements.txt
    pip install -e .

3. Run the API locally

    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Then open:
- http://localhost:8000/ for the UI
- http://localhost:8000/docs for API docs

---

## Deployment

This service is deployed on Render as a Docker web service.

Key configuration:
- Docker command:
  
    uvicorn api.app:app --host 0.0.0.0 --port $PORT

- Health check path: /healthz

---
## Author

Suraj Yarrapathruni
