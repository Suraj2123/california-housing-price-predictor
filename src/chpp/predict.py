from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from chpp.features import add_ratio_features

FEATURE_NAMES: list[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def make_feature_row(payload: Dict[str, float], feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """
    Convert a JSON payload into a (1, n_features) numpy row in the expected feature order.
    """
    try:
        row = np.array([[payload[f] for f in feature_names]], dtype=float)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(f"Missing feature: {missing}") from e
    return row


def make_feature_frame(payload: Dict[str, float]) -> pd.DataFrame:
    """
    Convert a JSON payload into a single-row DataFrame with engineered features.
    """
    base = {name: float(payload[name]) for name in FEATURE_NAMES}
    df = pd.DataFrame([base])
    return add_ratio_features(df)


def predict_one(model: Any, payload: Dict[str, float]) -> float:
    """
    Predict a single target value using a fitted sklearn-like model with .predict().
    """
    x = make_feature_frame(payload)
    pred = model.predict(x)[0]
    return float(pred)

