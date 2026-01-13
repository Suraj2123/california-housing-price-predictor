# California Housing Price Predictor

An end-to-end regression project built using scikit-learn’s California Housing dataset.
This project demonstrates how to structure, train, test, and serve a machine learning
model using production-style engineering practices.

## Project Highlights
- Modular ML pipeline (data loading, feature engineering, training, inference)
- Train / validation / test split
- Baseline → improved model workflow (extensible)
- Saved model artifacts and metrics
- FastAPI-based prediction service
- Automated CI with ruff (linting) and pytest
- Versioned release and documented model card

## Repository Structure
src/chpp/        # Core ML package  
api/             # FastAPI application  
tests/           # CI-safe smoke tests  
.github/         # GitHub Actions CI  

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
