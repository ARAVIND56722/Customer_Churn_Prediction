# Customer Churn Prediction & Risk Analytics System

An end-to-end Machine Learning pipeline that predicts whether a customer will churn based on historical behavioral and demographic data, identifies the critical drivers behind customer defection, and provides individual customer risk scoring and strategic retention recommendations.

---

## 📌 Project Overview

Customer churn is one of the most critical metrics in subscription and service businesses. This repository implements an end-to-end classification system designed to:
1. **Analyze Customer Behavior**: Comprehensive exploratory data analysis across tenure, contract structures, internet services, add-on packages, and payment methods.
2. **Benchmark Classification Models**: Evaluates multiple machine learning models (Logistic Regression, Random Forest, Gradient Boosting, Decision Tree) using 5-fold Stratified Cross-Validation and holdout test evaluation.
3. **Isolate Key Churn Drivers**: Quantifies risk elevators and retention anchors using Odds Ratios ($e^\beta$) and tree-based Permutation Importances.
4. **Deploy Real-Time Inference**: Provides both a CLI tool (`predict.py`) and an interactive web dashboard (`app.py`) for simulating individual customer profiles and batch CSV scoring.

---

## 🗂️ Project Directory Structure

```text
Customer_churn/
├── data/
│   └── telco_customer_churn.csv      # Historical customer dataset (7,043 records, 21 attributes)
├── src/
│   ├── __init__.py
│   ├── preprocess.py                 # Data cleaning, type conversion, and ColumnTransformer pipeline
│   ├── eda.py                        # Segment-level churn analytics and distribution metrics
│   ├── train.py                      # Multi-model training, 5-fold CV, and model serialization
│   └── explain.py                    # Feature importance, odds ratios, and retention playbooks
├── models/
│   ├── best_churn_model.joblib       # Serialized production pipeline (preprocessor + classifier)
│   ├── logistic_regression_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   └── gradient_boosting_pipeline.joblib
├── reports/
│   ├── eda_summary.json              # Cohort and segment churn metrics
│   ├── model_comparison.json         # Cross-validation and test set metrics across all models
│   └── feature_importance.json       # Top positive and negative churn drivers
├── predict.py                        # Standalone CLI prediction tool
├── app.py                            # Streamlit interactive risk simulator and analytics dashboard
├── generate_dataset.py               # Dataset generation and schema synthesis script
└── README.md                         # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment
Ensure Python 3.10+ is installed with `scikit-learn`, `pandas`, `numpy`, and `streamlit`:
```powershell
python -m pip install scikit-learn pandas numpy streamlit
```

### 2. Run the Full ML Pipeline
Execute EDA, model training, cross-validation, and explainability extraction:
```powershell
python -m src.eda; python -m src.train; python -m src.explain
```

### 3. Run Individual Customer Predictions (CLI)
Evaluate built-in risk benchmark profiles:
```powershell
python predict.py --profile all
```
Or test a specific profile (`high_risk`, `loyal_retained`, `moderate_risk`):
```powershell
python predict.py --profile high_risk
```

Batch score customer records from a CSV file:
```powershell
python predict.py --file data/telco_customer_churn.csv
```

### 4. Launch the Interactive Web Dashboard
Run the Streamlit application to explore data visualizations and run live churn simulations:
```powershell
streamlit run app.py
```

---

## 📊 Model Performance Summary

| Model | Test ROC-AUC | 5-Fold CV ROC-AUC | Recall (Churners) | Precision | Test Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Champion)** | **0.8913** | **0.8835 ± 0.008** | **84.97%** | 51.78% | 78.21% | 0.643 |
| **Random Forest** | 0.8853 | 0.8792 ± 0.009 | 83.44% | 52.21% | 78.50% | 0.642 |
| **Gradient Boosting** | 0.8835 | 0.8772 ± 0.009 | 62.27% | **70.00%** | **85.10%** | 0.659 |
| **Decision Tree** | 0.8827 | 0.8633 ± 0.006 | 81.29% | 55.56% | 80.62% | 0.660 |

> **Key takeaway**: Logistic Regression with balanced weighting achieved the top ranking ability (**0.8913 ROC-AUC**) and correctly caught **85% of churners** (277 out of 326), while Gradient Boosting achieved the highest overall accuracy (**85.1%**) and highest precision (**70.0%**).

---

## 🔍 Key Factors Identified

### 🚨 Top Churn Risk Drivers (Elevators)
1. **Fiber Optic Internet** (+315% higher odds of churn): High customer expectations and cost sensitivity; failure to resolve technical friction rapidly leads to defection.
2. **Electronic Check Payment** (+67.6% higher odds): Manual payment intervention reminds customers of recurring monthly bills.
3. **Month-to-Month Contracts** (40.0% churn rate vs 0.7% for 2-year contracts): Zero exit friction.
4. **Early Tenure (< 12 months)** (47.3% churn rate in the first year): Critical onboarding vulnerability window.

### 🛡️ Top Retention Anchors (Protectors)
1. **Two-Year & One-Year Contracts** (-95.9% and -85.8% reduction in churn odds): Strongest predictor of long-term retention.
2. **Long Tenure (> 24 months)** (-64.8% reduction in churn odds): Customers who pass the initial onboarding phase display high loyalty.
3. **Tech Support & Online Security** (-22% reduction in churn odds): Add-on services increase product stickiness and switching barriers.
4. **Automatic Billing (Credit Card / Bank Transfer)**: Frictionless payment reduces decision points.

