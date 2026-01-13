from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from chpp.data import load_raw_dataframe
from chpp.features import add_ratio_features, split_xy

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    val_size: float = 0.2  # portion of remaining after test split
    random_state: int = 42
    model_name: str = "hgb"  # "linear" or "hgb"


def _build_preprocess_pipeline(X):
    numeric_cols = list(X.columns)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def _build_model(model_name: str):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            random_state=42,
            max_depth=None,
            learning_rate=0.08,
            max_iter=350,
        )
    raise ValueError("model_name must be one of: ['linear', 'hgb']")


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> int:
    cfg = TrainConfig()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataframe()
    df = add_ratio_features(df)

    X, y = split_xy(df)

    # Split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    preprocessor = _build_preprocess_pipeline(X_train)
    model = _build_model(cfg.model_name)

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)

    val_pred = pipe.predict(X_val)
    test_pred = pipe.predict(X_test)

    metrics = {
        "model_name": cfg.model_name,
        "val": _metrics(y_val, val_pred),
        "test": _metrics(y_test, test_pred),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Save model + metrics
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(pipe, model_path)

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", model_path)
    print("Metrics:", json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
