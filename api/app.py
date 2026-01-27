from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

TEMPLATES_DIR = Path(__file__).parent / "templates"

@app.get("/", response_class=HTMLResponse)
def home():
    return (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    # TODO: replace with real values from your training artifacts/metrics.json
    return {
        "model_type": "your-model-here",
        "features": 8,
        "notes": "Fill this from metrics/artifacts so reviewers see governance + reproducibility."
    }

