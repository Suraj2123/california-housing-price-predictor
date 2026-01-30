from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd

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


def test_compare_locations(monkeypatch):
    monkeypatch.setattr(api_app, "get_model_state", lambda: _dummy_state())

    class DummyRegion:
        def __init__(self, region_id, name, lat, lon):
            self.region_id = region_id
            self.name = name
            self.lat = lat
            self.lon = lon

    monkeypatch.setattr(
        api_app,
        "get_regions",
        lambda: [DummyRegion("a", "Alpha", 1.0, 2.0), DummyRegion("b", "Beta", 3.0, 4.0)],
    )
    monkeypatch.setattr(
        api_app,
        "default_payload_for_region",
        lambda rid: {
            "MedInc": 5.0,
            "HouseAge": 25.0,
            "AveRooms": 5.5,
            "AveBedrms": 1.0,
            "Population": 1200.0,
            "AveOccup": 2.8,
            "Latitude": 34.05,
            "Longitude": -118.25,
        },
    )

    client = TestClient(api_app.app)
    response = client.get("/compare/locations?region_ids=a,b")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2


def test_trends_hpi(monkeypatch):
    points = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "value": [100.0, 101.0],
            "series_id": ["CSUSHPINSA", "CSUSHPINSA"],
        }
    )

    class DummySeries:
        def __init__(self):
            self.series_id = "CSUSHPINSA"
            self.region_id = "us"
            self.points = points

    monkeypatch.setattr(api_app, "load_hpi_series", lambda region_id: DummySeries())

    client = TestClient(api_app.app)
    response = client.get("/trends/hpi?region_id=us&max_points=24")
    assert response.status_code == 200
    data = response.json()
    assert data["series_id"] == "CSUSHPINSA"


def test_insights(monkeypatch):
    monkeypatch.setattr(api_app, "get_model_state", lambda: _dummy_state())

    class DummySeries:
        def __init__(self):
            self.series_id = "CSUSHPINSA"
            self.region_id = "us"
            self.points = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                    "value": [100.0, 101.0],
                    "series_id": ["CSUSHPINSA", "CSUSHPINSA"],
                }
            )

    monkeypatch.setattr(api_app, "load_hpi_series", lambda region_id: DummySeries())

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
    response = client.post("/insights?region_id=us", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data
