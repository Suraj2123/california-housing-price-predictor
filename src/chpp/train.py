from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from chpp.data import load_raw_dataframe
from chpp.evaluation.comparison import ModelResult, select_best_model, to_table
from chpp.evaluation.feature_importance import compute_feature_importance
from chpp.features import add_ratio_features, split_xy
from chpp.models.ensemble import WeightedEnsemble
from chpp.pipelines.preprocess import build_preprocess_pipeline
from chpp.pipelines.tuning import tune_hist_gbr

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    val_size: float = 0.2  # portion of remaining after test split
    random_state: int = 42
    model_name: str = "hgb"  # "linear" or "hgb"


def _build_model(model_name: str, random_state: int, params: dict | None = None):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "hgb":
        params = params or {"learning_rate": 0.08, "max_iter": 350, "max_depth": None}
        return HistGradientBoostingRegressor(
            random_state=random_state,
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_depth=params["max_depth"],
        )
    raise ValueError("model_name must be one of: ['linear', 'hgb']")


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _train_pipeline(model, X_train, y_train) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("prep", build_preprocess_pipeline(X_train)),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def _best_ensemble_weight(y_val, preds_a, preds_b) -> tuple[float, float]:
    best_weight = 0.5
    best_rmse = float("inf")
    for weight in np.linspace(0.0, 1.0, 11):
        blended = weight * preds_a + (1.0 - weight) * preds_b
        rmse = _metrics(y_val, blended)["rmse"]
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = float(weight)
    return best_weight, best_rmse


def train_model(cfg: TrainConfig | None = None) -> tuple[Pipeline, dict]:
    cfg = cfg or TrainConfig()

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

    feature_names = list(X_train.columns)

    linear_model = _build_model("linear", cfg.random_state)
    linear_pipe = _train_pipeline(linear_model, X_train, y_train)

    hgb_model = _build_model("hgb", cfg.random_state)
    hgb_pipe = _train_pipeline(hgb_model, X_train, y_train)

    tuning = tune_hist_gbr(X_train, y_train, X_val, y_val, cfg.random_state)
    tuned_params = tuning["best_params"]
    hgb_tuned = _build_model("hgb", cfg.random_state, tuned_params)
    hgb_tuned_pipe = _train_pipeline(hgb_tuned, X_train, y_train)

    candidates: list[ModelResult] = []
    for name, pipe, params in [
        ("linear", linear_pipe, {}),
        ("hgb", hgb_pipe, {"learning_rate": 0.08, "max_iter": 350, "max_depth": None}),
        ("hgb_tuned", hgb_tuned_pipe, tuned_params),
    ]:
        val_pred = pipe.predict(X_val)
        test_pred = pipe.predict(X_test)
        candidates.append(
            ModelResult(
                name=name,
                params=params,
                val=_metrics(y_val, val_pred),
                test=_metrics(y_test, test_pred),
            )
        )

    val_pred_linear = np.array(linear_pipe.predict(X_val))
    val_pred_hgb = np.array(hgb_tuned_pipe.predict(X_val))
    weight, _ = _best_ensemble_weight(y_val, val_pred_linear, val_pred_hgb)
    ensemble = WeightedEnsemble([linear_pipe, hgb_tuned_pipe], [weight, 1.0 - weight])
    ensemble_val_pred = np.array(ensemble.predict(X_val))
    ensemble_test_pred = np.array(ensemble.predict(X_test))
    ensemble_result = ModelResult(
        name="ensemble",
        params={"weight_linear": weight, "weight_hgb": 1.0 - weight},
        val=_metrics(y_val, ensemble_val_pred),
        test=_metrics(y_test, ensemble_test_pred),
    )
    candidates.append(ensemble_result)

    best = select_best_model(candidates)
    best_model: Pipeline | WeightedEnsemble
    if best.name == "ensemble":
        best_model = ensemble
    elif best.name == "hgb_tuned":
        best_model = hgb_tuned_pipe
    elif best.name == "hgb":
        best_model = hgb_pipe
    else:
        best_model = linear_pipe

    feature_importance = compute_feature_importance(
        best_model,
        X_val,
        y_val,
        feature_names=feature_names,
        random_state=cfg.random_state,
    )

    metrics = {
        "model_name": best.name,
        "val": best.val,
        "test": best.test,
        "comparison": to_table(candidates),
        "tuning": tuning,
        "ensemble": ensemble_result.params,
        "feature_importance": feature_importance,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "feature_names": feature_names,
    }

    return best_model, metrics


def save_artifacts(model: object, metrics: dict) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(model, model_path)

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if metrics.get("feature_importance"):
        with open(REPORTS_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
            json.dump(metrics["feature_importance"], f, indent=2)

    return model_path


def main() -> int:
    model, metrics = train_model()
    model_path = save_artifacts(model, metrics)

    print("Saved:", model_path)
    print("Metrics:", json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
