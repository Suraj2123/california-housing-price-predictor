from __future__ import annotations

from fastapi.testclient import TestClient

import api.app as api_app


class DummyModel:
    def predict(self, X):  # noqa: N803 - sklearn style uses X
        return [2.5 for _ in range(len(X))]


def _dummy_state() -> api_app.ModelState:
    return api_app.ModelState(
        model=DummyModel(),
        metrics={"test": {"rmse": 0.5, "mae": 0.4, "r2": 0.7}},
        error=None,
        trained_at="2026-01-01T00:00:00Z",
        source="tests",
        model_type="DummyModel",
    )


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(api_app, "get_model_state", lambda: _dummy_state())
    client = TestClient(api_app.app)

    payload = {
        "MedInc": 5.0,
        "HouseAge": 25.0,
        "AveRooms": 5.5,
        "AveBedrms": 1.0,
        "Population": 1200.0,
        "AveOccup": 2.8,
        "Latitude": 34.05,
        "Longitude": -118.25,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_hundreds_k"] == 2.5
    assert data["prediction_usd"] == 250000.0
    assert data["rmse_usd"] == 50000.0


def test_model_info(monkeypatch):
    monkeypatch.setattr(api_app, "get_model_state", lambda: _dummy_state())
    client = TestClient(api_app.app)

    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["model_type"] == "DummyModel"


def test_predict_batch(monkeypatch):
    monkeypatch.setattr(api_app, "get_model_state", lambda: _dummy_state())
    client = TestClient(api_app.app)

    payload = {
        "items": [
            {
                "MedInc": 5.0,
                "HouseAge": 25.0,
                "AveRooms": 5.5,
                "AveBedrms": 1.0,
                "Population": 1200.0,
                "AveOccup": 2.8,
                "Latitude": 34.05,
                "Longitude": -118.25,
            },
            {
                "MedInc": 6.2,
                "HouseAge": 18.0,
                "AveRooms": 4.8,
                "AveBedrms": 1.1,
                "Population": 900.0,
                "AveOccup": 2.5,
                "Latitude": 36.77,
                "Longitude": -119.42,
            },
        ]
    }

    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["prediction_usd"] == 250000.0
