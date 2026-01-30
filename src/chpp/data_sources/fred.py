from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import io
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
DEFAULT_SERIES_ID = "CSUSHPINSA"


@dataclass(frozen=True)
class HpiSeries:
    series_id: str
    region_id: str
    points: pd.DataFrame


SERIES_BY_REGION = {
    "us": "CSUSHPINSA",
    "ca": "CASTHPI",
}


def resolve_series_id(region_id: str | None) -> str:
    if not region_id:
        return DEFAULT_SERIES_ID
    return SERIES_BY_REGION.get(region_id, DEFAULT_SERIES_ID)


def _cache_path(cache_dir: Path, series_id: str) -> Path:
    return cache_dir / f"hpi_{series_id}.csv"


def _is_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
    return datetime.utcnow() - mtime < timedelta(hours=max_age_hours)


def _fetch_series(series_id: str, timeout_seconds: int = 10) -> pd.DataFrame:
    url = FRED_GRAPH_URL.format(series_id=series_id)
    resp = requests.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df = df[df["value"] != "."].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].astype(float)
    df["series_id"] = series_id
    return df


def load_hpi_series(
    region_id: str | None = None,
    cache_dir: Path | str = Path("artifacts") / "trends",
    max_age_hours: int = 24,
) -> HpiSeries:
    series_id = resolve_series_id(region_id)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(cache_dir, series_id)

    if _is_fresh(cache_path, max_age_hours):
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        return HpiSeries(series_id=series_id, region_id=region_id or "us", points=cached)

    try:
        df = _fetch_series(series_id)
        df.to_csv(cache_path, index=False)
        return HpiSeries(series_id=series_id, region_id=region_id or "us", points=df)
    except Exception:
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            return HpiSeries(series_id=series_id, region_id=region_id or "us", points=cached)
        raise


def summarize_changes(points: pd.DataFrame, windows: Iterable[int] = (12, 60)) -> dict:
    points = points.sort_values("date")
    latest = points.iloc[-1]["value"]
    summary = {"latest": float(latest)}
    for window in windows:
        if len(points) <= window:
            continue
        prev = points.iloc[-(window + 1)]["value"]
        summary[f"change_{window}m_pct"] = float((latest - prev) / prev * 100.0)
    return summary
