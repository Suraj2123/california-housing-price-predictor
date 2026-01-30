from __future__ import annotations

from typing import List

import numpy as np
from sklearn.inspection import permutation_importance


def compute_feature_importance(
    model,
    X_val,
    y_val,
    feature_names: List[str],
    random_state: int,
    n_repeats: int = 5,
    top_k: int = 10,
) -> dict:
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )
    importances = result.importances_mean
    order = np.argsort(importances)[::-1]
    top = [
        {"feature": feature_names[idx], "importance": float(importances[idx])}
        for idx in order[:top_k]
    ]
    return {
        "top_features": top,
        "n_repeats": n_repeats,
    }
