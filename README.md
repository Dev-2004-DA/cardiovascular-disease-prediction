# Cardiovascular Disease Prediction — ML Classification

**Project completed:** February 2026  
**Tools:** Python · Scikit-learn · Pandas · Matplotlib · Seaborn  
**Domain:** Machine Learning · Healthcare · Binary Classification

---

## Overview

An end-to-end machine learning pipeline built on a dataset of **70,000 medical records**  
to predict the presence of cardiovascular disease.

The project compares 10+ classification algorithms and focuses heavily on improving  
**disease recall** — because in a medical context, missing a positive case (false negative)  
is far more costly than a false alarm.

**Key result:** Recall improved from **69% to 85%** through threshold tuning and  
feature importance analysis, using Random Forest as the final model.

---

## Problem Statement

Cardiovascular disease is one of the leading causes of death globally. Early prediction  
using patient health metrics can significantly improve outcomes. The challenge in this  
dataset is the **class imbalance** and the need to prioritize recall over accuracy —  
a model that misses sick patients is dangerous, even if its overall accuracy looks good.

---

## Methodology

### 1. Data Preparation
- Dataset: 70,000 patient records with features including age, height, weight,  
  blood pressure, cholesterol, glucose levels, smoking, alcohol, and physical activity
- Applied stratified train-test split to preserve class distribution
- No data leakage — all preprocessing fitted on training set only

### 2. Model Comparison
Evaluated 10+ classifiers including:
- Logistic Regression
- Decision Tree
- Random Forest ✅ (selected)
- Gradient Boosting
- XGBoost
- SVM
- K-Nearest Neighbors
- Naive Bayes

### 3. Model Selection
- Selected **Random Forest** (~76% cross-validation accuracy)
- Validated using **OOB (Out-of-Bag) score** — a built-in unbiased estimator
- Applied **5-fold stratified cross-validation** for bias-variance diagnostics

### 4. Threshold Tuning & Recall Improvement

| Metric | Before Tuning | After Tuning |
|---|---|---|
| Recall (Disease) | 69% | **85%** |
| ROC-AUC | Baseline | Improved |
| Threshold | 0.50 (default) | Optimized via ROC curve |

Plotted full **Precision-Recall curve** and **ROC-AUC curve** to identify the optimal  
classification threshold for the medical use case.

### 5. Feature Importance
- Ranked all features by Random Forest importance scores
- Top predictors: age, systolic blood pressure, cholesterol level
- Confirmed results align with medical domain knowledge

---

## Key Findings

- Default 0.5 threshold is wrong for medical classification — always tune it
- OOB validation gives reliable estimates without needing a separate validation set
- Random Forest significantly outperformed linear models due to feature interactions
- Recall is the right metric here — optimizing for accuracy alone would miss 31% of sick patients

---

## Repository Structure

```
cardiovascular-disease-prediction/
├── cardiovascular_disease_prediction.ipynb   # Full analysis notebook
├── README.md
```

> Dataset sourced from Kaggle — download and place in the root folder before running.

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/Dev-2004-DA/cardiovascular-disease-prediction.git
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

3. Download the dataset from Kaggle and place it in the project folder

4. Open the notebook
```bash
jupyter notebook cardiovascular_disease_prediction.ipynb
```

---

## Skills Demonstrated

`Binary Classification` `Random Forest` `Threshold Tuning` `ROC-AUC`  
`Precision-Recall` `Cross-Validation` `OOB Validation` `Feature Importance`  
`Stratified Split` `Scikit-learn` `Python` `Healthcare Analytics`

---

*Part of my Data Analytics portfolio — [github.com/Dev-2004-DA](https://github.com/Dev-2004-DA)*
