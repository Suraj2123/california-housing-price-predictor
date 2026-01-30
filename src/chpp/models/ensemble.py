from __future__ import annotations

from typing import Iterable, List

import numpy as np


class WeightedEnsemble:
    def __init__(self, estimators: Iterable, weights: Iterable[float]):
        self.estimators = list(estimators)
        self.weights = np.array(list(weights), dtype=float)
        if len(self.estimators) != len(self.weights):
            raise ValueError("Estimators and weights must have the same length.")
        if np.isclose(self.weights.sum(), 0.0):
            raise ValueError("Weights must not sum to zero.")
        self.weights = self.weights / self.weights.sum()

    def predict(self, X) -> List[float]:
        preds = np.zeros(len(X), dtype=float)
        for estimator, weight in zip(self.estimators, self.weights):
            preds += weight * estimator.predict(X)
        return preds.tolist()
