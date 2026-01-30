from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelResult:
    name: str
    params: Dict[str, float | int | None]
    val: Dict[str, float]
    test: Dict[str, float]


def select_best_model(results: List[ModelResult]) -> ModelResult:
    if not results:
        raise ValueError("No model results to compare.")
    return min(results, key=lambda item: item.val["rmse"])


def to_table(results: List[ModelResult]) -> List[Dict[str, float | str]]:
    table = []
    for result in results:
        row = {
            "model": result.name,
            "val_rmse": result.val["rmse"],
            "val_mae": result.val["mae"],
            "val_r2": result.val["r2"],
            "test_rmse": result.test["rmse"],
            "test_mae": result.test["mae"],
            "test_r2": result.test["r2"],
        }
        table.append(row)
    return table
