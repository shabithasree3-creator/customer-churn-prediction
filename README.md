# Customer Churn Prediction — Machine Learning Portfolio Project

## Overview
This project demonstrates an end-to-end classification workflow for predicting customer churn. It compares Logistic Regression, Decision Tree, and Random Forest models and evaluates them using precision, recall, F1-score, ROC-AUC, and confusion matrices.

**Data note:** The included workflow generates a reproducible synthetic customer dataset for demonstration. Results from synthetic data are not evidence of real-world business performance. For a stronger portfolio, replace it with a documented public churn dataset and rerun the analysis.

## Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Workflow
1. Generate/load customer records
2. Explore and prepare features
3. Split into train and test sets
4. Train models in Scikit-learn pipelines
5. Compare classification metrics
6. Save a model comparison table and confusion-matrix plot
7. Discuss limitations and possible business actions

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Outputs are written to `outputs/`.

## How to improve this project
- Use a real, cited telecom/customer churn dataset.
- Add cross-validation and hyperparameter tuning.
- Test for class imbalance and tune the decision threshold.
- Add explainability and calibration checks.
- Deploy a trained model through a small REST API only after validating it.

## Resume use
After you run this project, review the generated metrics and be ready to explain the train/test split, leakage prevention, precision vs. recall, and why you chose the final model. Do not claim real-world impact or specific performance before validating on a real dataset.
