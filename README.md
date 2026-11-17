# Credit Card Behaviour Score

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-archived-lightgrey)

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

1. **EDA**: class distribution (`bad_flag`: 95,434 good vs 1,372 bad, about 1.4% positives).
2. **Feature selection**: only the integer-typed columns are kept (`select_dtypes(int, float)` includes `int`
   and *excludes* `float`), and the constant column `onus_attribute_28` is dropped. This leaves 25 inputs
   (`account_number` plus 24 `onus_attribute_*` columns); no imputation is done because the float columns
   that carry the missing values are excluded.
3. **Baseline**: plain Logistic Regression on a 70/30 split predicts only the majority class
   (recall 0.00 on defaulters).
4. **Class imbalance**: SMOTE is applied to the full development set, then the oversampled data is split 70/30.
5. **Models compared**: Logistic Regression (`class_weight="balanced"`) and Random Forest (default params).
6. **Threshold tuning**: the F1-optimal threshold on the precision-recall curve (`0.4718`) is computed.
7. **Output**: the Logistic Regression model is saved to `behaviour_score_LR.joblib`, the threshold to
   `model_threshold.json`, and validation-set probabilities to `validation_predictions.csv` (not committed).

## Results

Numbers as printed in the notebook (test split = 30% of the **SMOTE-oversampled** data, 57,261 rows):

| Model | Accuracy | Precision (bad) | Recall (bad) | F1 (bad) | ROC-AUC |
|-------|---------:|----------------:|-------------:|---------:|--------:|
| Logistic Regression, no resampling (original 30% split) | 0.99 | 0.00 | 0.00 | 0.00 | – |
| Logistic Regression, balanced + SMOTE (**saved model**) | 0.69 | 0.69 | 0.70 | 0.69 | 0.75 |
| Random Forest + SMOTE | 0.92 | 0.91 | 0.94 | 0.92 | 0.97 |
| Random Forest + SMOTE, threshold 0.47 | 0.92 | 0.91 | 0.94 | 0.92 | 0.97 |

<details>
<summary>Known limitations (read before reusing these numbers)</summary>

- **SMOTE before the split**: oversampling is done on the whole dataset before `train_test_split`, so the test
  set contains synthetic minority samples interpolated from training points. The scores above are optimistic
  and are not an estimate of performance on real, imbalanced data.
- **Threshold / model mismatch**: `0.4718` was derived from the Random Forest's precision-recall curve, but
  the persisted model is the Logistic Regression.
- **`account_number` is used as a feature**, which is an identifier and should be excluded.
- Only 24 of roughly 1,200 behavioural attributes are used (the float-typed bureau and transaction features
  are dropped by the dtype filter).

A sounder setup would split first, oversample (or class-weight) only the training fold, use the full feature
set with imputation, and report PR-AUC on an untouched, imbalanced test set.

</details>

## Files

```
Credit-Card-Behaviour-Score/
├── behavior_score.ipynb        # End-to-end notebook (EDA → train → evaluate → predict)
├── behaviour_score_LR.joblib   # Trained Logistic Regression model
├── model_threshold.json        # {"optimal_threshold": 0.4718}
├── requirements.txt
├── LICENSE
└── README.md
```

## Stack

Python, pandas, NumPy, scikit-learn (Logistic Regression, Random Forest), imbalanced-learn (SMOTE), matplotlib, joblib.

## How to use the saved model

```python
import joblib, json, pandas as pd

model = joblib.load("behaviour_score_LR.joblib")
threshold = json.load(open("model_threshold.json"))["optimal_threshold"]

df_val = pd.read_csv("validation_data_to_be_shared.csv")
X = df_val[model.feature_names_in_]          # the 25 columns the model was trained on
proba = model.predict_proba(X)[:, 1]
predicted = (proba >= threshold).astype(int)
```

## How to reproduce

1. Download `Dev_data_to_be_shared.csv` and `validation_data_to_be_shared.csv` from the Kaggle dataset above and place them in the project root.
2. Install dependencies: `pip install -r requirements.txt`
3. Open `behavior_score.ipynb` and run all cells.

## License

Code is released under the [MIT License](LICENSE). The dataset comes from the IIT Bombay case study and is not
covered by this licence.
