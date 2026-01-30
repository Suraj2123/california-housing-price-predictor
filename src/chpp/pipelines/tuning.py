from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from chpp.pipelines.preprocess import build_preprocess_pipeline


@dataclass(frozen=True)
class TuningResult:
    params: Dict[str, float | int | None]
    rmse: float


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def tune_hist_gbr(
    X_train,
    y_train,
    X_val,
    y_val,
    random_state: int,
    param_grid: List[Dict[str, float | int | None]] | None = None,
) -> dict:
    grid = param_grid or [
        {"learning_rate": 0.05, "max_iter": 250, "max_depth": None},
        {"learning_rate": 0.08, "max_iter": 350, "max_depth": None},
        {"learning_rate": 0.1, "max_iter": 350, "max_depth": 6},
    ]

    results: List[TuningResult] = []
    best_params = grid[0]
    best_rmse = float("inf")

    for params in grid:
        model = HistGradientBoostingRegressor(
            random_state=random_state,
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_depth=params["max_depth"],
        )
        pipe = Pipeline(
            steps=[
                ("prep", build_preprocess_pipeline(X_train)),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        rmse = _rmse(y_val, preds)
        results.append(TuningResult(params=params, rmse=rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    return {
        "best_params": best_params,
        "results": [
            {"params": result.params, "rmse": result.rmse} for result in results
        ],
    }
