# Credit Card Behaviour Score

## Overview
This project is based on the IIT Bombay case study for **Credit Card Behaviour Scoring**. The objective is to develop a predictive model that estimates the probability of credit card customers defaulting. The model will be used for risk management and portfolio optimization.

## Problem Statement
Bank A has issued credit cards to various customers and now seeks to assess their default risk. The goal is to develop a **Behaviour Score** that predicts the likelihood of a customer defaulting based on historical and transactional data.

## Dataset
The project involves two datasets:

1. **Development Data** (`Dev_data_to_be_shared.zip`):  
   - Contains **96,806** credit card records.  
   - Includes independent variables related to:
     - **Credit limits** (`onus_attributes_*`)
     - **Transaction data** (`transaction_attribute_*`)
     - **Bureau tradelines** (`bureau_*`)
     - **Bureau enquiries** (`bureau_enquiry_*`)
   - A **bad_flag** column indicates whether a customer has defaulted (`1 = default, 0 = no default`).

2. **Validation Data** (`validation_data_to_be_shared.zip`):  
   - Contains **41,792** records.  
   - Has the same features as the development dataset but **without the bad_flag**.

## Objective
1. Train a **predictive model** on the **development dataset** to estimate default probability.  
2. Apply the trained model to **validation data** to generate probabilities for each customer.  
3. Submit a CSV file with two columns:
   - `account_number` (Primary key)
   - `predicted_probability` (Estimated probability of default)
4. Document the approach, algorithms, insights, and model evaluation metrics.

## Approach
1. **Data Preprocessing**
   - Handle missing values.
   - Feature engineering and selection.
   - Standardization/Normalization.

2. **Exploratory Data Analysis (EDA)**
   - Understanding feature distributions.
   - Identifying correlations with `bad_flag`.
   - Checking data imbalance.

3. **Model Development**
   - Train different models (e.g., Logistic Regression, Decision Trees, Random Forest, XGBoost).
   - Use **AUC-ROC, Precision-Recall, and F1-score** for evaluation.
   - Select the best model based on performance.

4. **Prediction & Validation**
   - Apply the trained model to validation data.
   - Generate and save predicted probabilities.

## Metrics Used
- **AUC-ROC Curve**
- **Precision-Recall Curve**
- **F1 Score**
- **Log Loss**
- **Gini Coefficient**

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn)
- Jupyter Notebook
- GitHub

## How to Run
1. Download the dataset from the provided link.
2. Run `data_preprocessing.py` to clean and preprocess data.
3. Train models using `train_model.py`.
4. Predict on validation data using `predict.py`.
5. Save the final submission file.


## Dataset Access
The dataset can be accessed [here](https://www.kaggle.com/datasets/rupeshbharambe/credit-card-behaviour-score/data).

## Author
**[Rupesh Bharambe](https://www.linkedin.com/in/rupesh-bharambe-056a12257/)**

