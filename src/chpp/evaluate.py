from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from chpp.features import add_features, split_xy

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

MODEL_PATH = MODELS_DIR / "model.joblib"


def load_dataframe() -> pd.DataFrame:
    ds = fetch_california_housing(as_frame=True)
    return ds.frame.copy()


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first: python -m chpp.train"
        )

    model = joblib.load(MODEL_PATH)

    # Recreate the same split as training (same random state)
    df = add_features(load_dataframe())
    X, y = split_xy(df)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Predict on test set
    y_pred = model.predict(X_test)

    # Save detailed predictions
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
    pred_path = REPORTS_DIR / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Residual plot
    plt.figure()
    plt.scatter(pred_df["y_pred"], pred_df["residual"], s=8)
    plt.axhline(0)
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (true - pred)")
    plot_path = REPORTS_DIR / "residuals_vs_pred.png"
    plt.savefig(plot_path, bbox_inches="tight")

    # Error buckets: group by predicted value ranges
    pred_df["pred_bucket"] = pd.qcut(pred_df["y_pred"], q=5, duplicates="drop")
    bucket = (
        pred_df.groupby("pred_bucket")
        .agg(
            count=("residual", "size"),
            mean_abs_error=("residual", lambda s: float(np.mean(np.abs(s)))),
            mean_residual=("residual", lambda s: float(np.mean(s))),
        )
        .reset_index()
    )
    bucket_path = REPORTS_DIR / "error_buckets.csv"
    bucket.to_csv(bucket_path, index=False)

    summary = {
        "artifacts": {
            "predictions_csv": str(pred_path),
            "residual_plot": str(plot_path),
            "error_buckets_csv": str(bucket_path),
        }
    }
    with open(REPORTS_DIR / "evaluation_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote evaluation artifacts:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
