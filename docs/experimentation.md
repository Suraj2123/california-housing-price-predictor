# Experimentation Framework

This project documents a lightweight A/B testing framework to connect model improvements to business outcomes.

## Example hypothesis

Switching the default recommender from baseline to tuned ensemble improves user satisfaction without lowering coverage.

## Experiment design

- **Variants**: Baseline model vs. tuned ensemble
- **Population split**: 50/50 random assignment
- **Duration**: Minimum 2 weeks to capture seasonal effects
- **Primary metric**: Conversion proxy (user clicks or saves)
- **Secondary metrics**: Prediction confidence display engagement, API latency, coverage

## Metrics to capture

- **CTR proxy**: % of prediction results that lead to a follow-up action
- **Engagement**: time on dashboard, number of comparisons performed
- **Coverage**: % of inputs that receive predictions without fallback
- **Latency**: P95 response time for `/predict` and `/predict-batch`

## Guardrails

- Revert if latency exceeds 2x baseline
- Revert if error rate exceeds 1%

## Analysis checklist

- Confirm sample ratio mismatch is within tolerance
- Compare primary metrics with confidence intervals
- Audit data drift during experiment window
