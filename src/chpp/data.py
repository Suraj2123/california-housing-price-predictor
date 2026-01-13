from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_raw_dataframe() -> pd.DataFrame:
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame.copy()
    return df
