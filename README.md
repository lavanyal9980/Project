# Employee Attrition Prediction using Logistic Regression

This project uses a Logistic Regression model to predict employee attrition based on various HR-related features. The goal is to proactively identify employees at risk of leaving and support retention strategies using data-driven insights.

# Project Structure
- script.py: Used for data preprocessing and data visualization
- script2.py: Used to build a Logistics Regression model and train it
- script4.py: Used to perform SHAP Value Analysis

# Dataset

- **Source**: IBM HR Analytics Employee Attrition & Performance dataset
- **Rows**: ~1,470 employees
- **Target Variable**: `Attrition` (Yes/No)

## ⚙️ Features & Workflow

# **Data Preprocessing**
- Dropped irrelevant columns: `EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`
- Label Encoding of categorical features
- Standard scaling of numerical variables

# **Feature Selection**
- Applied Recursive Feature Elimination (RFE) to select top 15 most predictive features

# **Modeling**
- Used Logistic Regression with GridSearchCV for hyperparameter tuning
  - Penalty: `l1`, `l2`
  - C values: `[0.01, 0.1, 1, 10]`
  - Cross-validation: 5 folds
- Balanced class weights to address class imbalance

# **Evaluation Metrics**
- **Accuracy**: 73.47%
- **Precision (Attrition = Yes)**: 36%
- **Recall (Attrition = Yes)**: 81%
- Confusion Matrix and Classification Report included

# Key Findings

- The model performs well in **recall**, identifying the majority of employees likely to leave.
- However, **precision is low**, leading to many false positives.
- This model is useful for early interventions, but should be paired with qualitative insights for final decisions.

