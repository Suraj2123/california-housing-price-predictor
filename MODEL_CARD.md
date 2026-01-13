# Model Card: California Housing Price Predictor

## Model Overview
This project predicts median house value in California census districts using
the scikit-learn California Housing dataset. The goal is to demonstrate an
end-to-end, reproducible machine learning workflow rather than maximize leaderboard performance.

## Dataset
- Source: scikit-learn `fetch_california_housing`
- Size: ~20,640 rows
- Target: `MedHouseVal` (median house value in hundreds of thousands USD)

### Features
- Median income
- House age
- Average rooms and bedrooms
- Population and average occupancy
- Geographic coordinates (latitude, longitude)
- Engineered features:
  - Rooms per bedroom
  - Capped average occupancy

## Modeling Approach
- Data split into train / validation / test sets
- Baseline linear regression considered
- Final model: Histogram Gradient Boosting Regressor
- Median imputation and standard scaling applied via sklearn pipelines

## Evaluation Metrics
- RMSE
- MAE
- R²

Metrics are evaluated on a held-out test set to avoid leakage.

## Limitations
- Dataset represents historical census data and may not reflect current market conditions
- Does not incorporate temporal dynamics or macroeconomic factors
- Geographic features may encode socio-economic bias

## Intended Use
This model is intended for educational and demonstration purposes only and should
not be used for real estate pricing decisions.
