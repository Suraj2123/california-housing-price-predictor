from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_training_pipeline_smoke(tmp_path):
    subprocess.check_call([sys.executable, "-m", "chpp.train"])

    metrics_path = Path("artifacts/reports/metrics.json")
    assert metrics_path.exists()

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "test" in data and "rmse" in data["test"]
