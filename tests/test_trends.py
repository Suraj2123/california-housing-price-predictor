from __future__ import annotations

from pathlib import Path

import pandas as pd

from chpp.data_sources.fred import load_hpi_series, summarize_changes


class DummyResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


def test_load_hpi_series_caches(monkeypatch, tmp_path: Path):
    csv = "DATE,CSUSHPINSA\n2020-01-01,100\n2020-02-01,101\n"

    def fake_get(*_args, **_kwargs):
        return DummyResponse(csv)

    monkeypatch.setattr("chpp.data_sources.fred.requests.get", fake_get)

    series = load_hpi_series(region_id="us", cache_dir=tmp_path, max_age_hours=0)
    assert series.series_id == "CSUSHPINSA"
    assert len(series.points) == 2

    cached = load_hpi_series(region_id="us", cache_dir=tmp_path, max_age_hours=9999)
    assert len(cached.points) == 2


def test_summarize_changes():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            "value": [100.0, 110.0, 121.0],
        }
    )
    summary = summarize_changes(df, windows=(1,))
    assert round(summary["change_1m_pct"], 2) == 10.0
