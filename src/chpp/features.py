from __future__ import annotations

import pandas as pd


TARGET_COL = "MedHouseVal"


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple, explainable ratio features commonly used for this dataset.
    Keeps it "portfolio-real": small but meaningful feature engineering.
    """
    out = df.copy()

    # Avoid division by zero using replace(0, 1) style safeguard
    households = out["HouseAge"] * 0 + 1  # dummy default if not present (won't be used)
    if "Households" in out.columns:
        households = out["Households"].replace(0, 1)

    if "AveRooms" in out.columns and "AveBedrms" in out.columns:
        # Sometimes derived features are already averages. We still create a ratio feature.
        out["RoomsPerBed"] = out["AveRooms"] / out["AveBedrms"].replace(0, 1)

    if "AveOccup" in out.columns:
        # Occupancy is already average, but we can create a capped version to reduce outlier impact
        out["AveOccupCapped"] = out["AveOccup"].clip(upper=10)

    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
