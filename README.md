# Credit Card Behaviour Score

A predictive model that estimates the probability of credit-card customer default. Built on the **IIT Bombay Behaviour Scoring case study** dataset.

## Problem

Bank A has issued credit cards to ~96K customers and needs to assess each customer's likelihood of defaulting. The objective is to produce, for each account in a held-out validation set, a `predicted_probability` of default that can be used downstream for risk management and portfolio optimization.

## Dataset

| File | Rows | Role |
|------|-----:|------|
| `Dev_data_to_be_shared.csv` | 96,806 | Training (with `bad_flag` label) |
| `validation_data_to_be_shared.csv` | 41,792 | Held-out scoring set (no label) |

Feature groups:
- `onus_attributes_*` — credit limit / on-us behaviour
- `transaction_attribute_*` — transaction patterns
- `bureau_*` — bureau tradelines
- `bureau_enquiry_*` — bureau enquiries

Dataset link: [Kaggle - Credit Card Behaviour Score](https://www.kaggle.com/datasets/rupeshbharambe/credit-card-behaviour-score/data).

## Approach

1. **EDA** — class distribution, feature distributions, correlation with `bad_flag`.
2. **Preprocessing** — missing-value handling, train/test split.
3. **Class imbalance** — SMOTE oversampling on the training set.
4. **Models compared** — Logistic Regression (baseline + class-weight `balanced`) and Random Forest.
5. **Threshold tuning** — used the precision-recall curve to pick an operating threshold (`0.4718`) instead of the default `0.5`, optimizing for recall on the minority class.
6. **Final model** — Logistic Regression saved to `behaviour_score_LR.joblib` with the chosen threshold in `model_threshold.json`.

## Files

```
Credit-Card-Behaviour-Score/
├── behavior_score.ipynb        # End-to-end notebook (EDA → train → evaluate → predict)
├── behaviour_score_LR.joblib   # Trained Logistic Regression model
├── model_threshold.json        # {"optimal_threshold": 0.4718}
└── README.md
```

## Metrics tracked

- ROC-AUC, Precision, Recall, F1
- Confusion matrix at default and tuned thresholds
- Precision-Recall and ROC curves

## Stack

Python, pandas, NumPy, scikit-learn (Logistic Regression, Random Forest), imbalanced-learn (SMOTE), matplotlib, joblib.

## How to use the saved model

```python
import joblib, json, pandas as pd

model = joblib.load("behaviour_score_LR.joblib")
threshold = json.load(open("model_threshold.json"))["optimal_threshold"]

X = pd.read_csv("validation_data_to_be_shared.csv")
proba = model.predict_proba(X)[:, 1]
predicted = (proba >= threshold).astype(int)
```

## How to reproduce

1. Download `Dev_data_to_be_shared.csv` and `validation_data_to_be_shared.csv` from the Kaggle dataset above and place them in the project root.
2. Open `behavior_score.ipynb` and run all cells.
