# California Housing Price Predictor (sklearn dataset)

End-to-end regression project using scikit-learn’s built-in California Housing dataset:
- feature engineering
- train/val/test split
- baseline vs improved model path (expandable)
- saved artifacts + metrics
- FastAPI prediction endpoint
- CI with ruff + pytest

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

python -m chpp.train
uvicorn api.app:app --reload
