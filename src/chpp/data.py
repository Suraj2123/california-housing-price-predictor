from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_raw_dataframe() -> pd.DataFrame:
    """
    Loads the California Housing dataset from scikit-learn and returns a DataFrame
    with the target column included as 'MedHouseVal'.

    No external downloads required beyond sklearn's dataset fetch.
    """
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame.copy()
    # ds.frame already includes target as 'MedHouseVal' for sklearn's dataset
    return df
