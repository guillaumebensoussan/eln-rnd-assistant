# Evaluation Report

Train rows: 90 · Test rows: 30 · random_state=42

## Regression — target `yield_pct`

| Model | R2 | MAE |
|---|---|---|
| LinearRegression | 0.8043 | 4.7625 |
| RandomForestRegressor | 0.6817 | 6.1497 |

Best: **LinearRegression**

## Classification — target `qc_pass`

| Model | Accuracy | F1 |
|---|---|---|
| LogisticRegression | 0.8667 | 0.7143 |
| RandomForestClassifier | 0.8333 | 0.4444 |

Best: **LogisticRegression**
