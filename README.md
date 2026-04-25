# cardiovascular-disease-prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellowgreen?logo=scikit-learn) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A machine learning classification project to predict cardiovascular disease risk in ~70,000 patients using ensemble methods, threshold optimisation, and clinical decision support analysis.

---

## Project Overview

This project builds and evaluates **11 classification models** to predict the presence or absence of cardiovascular disease (CVD). The goal is both predictive accuracy and **clinical utility** — specifically minimising missed disease cases (false negatives), which carry greater patient risk than false positives.

The final model — a **Random Forest Classifier** — was selected for its stable precision (~75–76%), controlled variance, and interpretable feature importance. Threshold analysis was then applied to calibrate the model for real-world clinical deployment scenarios.

---

## Dataset

- **Source:** [Kaggle — Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/akshatshaw7/cardiovascular-disease-dataset)
- **Records:** ~70,000 patients
- **Features:** 11 (objective, examination, subjective)
- **Target:** `cardio` — binary (0 = No CVD, 1 = CVD Present)
- **Class balance:** ~50/50 — no sampling adjustment required

| Feature | Category | Description |
|---|---|---|
| `age_yr` | Objective | Age in years (converted from days) |
| `gender` | Objective | 0 = Female, 1 = Male |
| `BMI_kg/m2` | Engineered | Derived from height & weight |
| `ap_hi` | Examination | Systolic blood pressure (mmHg) |
| `ap_lo` | Examination | Diastolic blood pressure (mmHg) |
| `cholesterol` | Examination | 1: Normal · 2: Above normal · 3: Well above normal |
| `gluc` | Examination | 1: Normal · 2: Above normal · 3: Well above normal |
| `smoke` | Subjective | Binary — smoker status |
| `alco` | Subjective | Binary — alcohol intake |
| `active` | Subjective | Binary — physical activity |

---

## Data Cleaning Highlights

Blood pressure columns (`ap_hi`, `ap_lo`) required a **sequential rule-based cleaning pipeline**:

- Typo correction (e.g., `ap_lo = 1000` → `100`)
- Removal of zero and physiologically impossible values
- Sign error correction (negative → positive)
- Division of extreme values (>10,000 and >1,000) by 10 or 100
- Final ceiling filter: removed all rows with BP > 300 mmHg
- BMI validated in range 10–100 kg/m²
- 7 missing values and duplicate rows removed

**Final cleaned dataset: ~68,000+ records**

---

## Models Evaluated

| # | Model | CV Precision | Notes |
|---|---|---|---|
| 1 | Logistic Regression (Elastic Net) | ~72% | Linear boundary insufficient |
| 2 | LR + Polynomial Features (degree 2) | ~72% | No improvement over base LR |
| 3 | Decision Tree | ~71% | Higher variance than ensembles |
| 4 | **Random Forest** ✅ | **~75–76%** | **Final model selected** |
| 5 | Categorical Naïve Bayes | ~76% | Limited by independence assumption |
| 6 | SVM (RBF, C=10, γ=0.1) | ~72% | ~25 min/fit — impractical to tune |
| 7 | Gradient Boosting | ~72% | CV variance dropped from 79% train |
| 8 | KNN (k=41–107, StandardScaler) | ~76% | Slower inference at scale |
| 9 | Bagging (SVM base, 10 estimators) | ~78% | No improvement over RF |
| 10 | Voting Classifier (soft, 8 models) | ~73% | No improvement over RF |
| 11 | PCA (4 components) + SVM | ~72% | Dimensionality reduction didn't help |

---

## Final Model — Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf1 = RandomForestClassifier(
    max_depth        = 10,
    max_features     = 4,
    min_samples_leaf = 45,
    min_samples_split= 30,
    n_estimators     = 200,
    oob_score        = True,
    random_state     = 45
)
```

| Metric | Value |
|---|---|
| Cross-validated Precision | ~75–76% |
| OOB Score | ~73% |
| Test Accuracy | ~74–76% |

---

## Feature Importance

| Rank | Feature | Importance | Insight |
|---|---|---|---|
| 1 | `ap_hi` | >50% | Dominant predictor — hypertension signal |
| 2 | `ap_lo` | ~20% | Complements systolic reading |
| 3 | `age_yr` | ~10% | Well-established CVD risk factor |
| 4 | `cholesterol` | ~8% | Moderate impact |
| 5 | `BMI_kg/m2` | ~5% | Engineered feature |
| — | `smoke/alco/active` | ~1–2% each | Low signal in cross-sectional data |

> Blood pressure features together contribute **~70% of total importance** — consistent with clinical knowledge that hypertension is the leading modifiable CVD risk factor.

---

## Threshold Analysis

At default threshold (0.50), the model misses **31% of actual disease cases** — clinically unacceptable. Lower thresholds were evaluated:

| Threshold | Precision | Recall (Disease) | Use Case |
|---|---|---|---|
| 0.50 (default) | 75% | 69% | Balanced — default behaviour |
| 0.40 | 69% | 80% | Fewer missed cases |
| 0.35 | 66% | 85% | High recall — broad screening |
| 0.30 | 62% | 89% | Maximum recall — population screening |

> **Recommended deployment:** Threshold **0.30–0.35** for population screening. Threshold **0.40–0.50** for resource-constrained diagnostic triage.

---

## BP Discretization Experiment

Systolic and diastolic BP were binned into clinical hypertension categories and tested against the continuous version:

| Model | Continuous CV Precision | Discretized CV Precision |
|---|---|---|
| Random Forest | 75.73% | 75.58% |
| Gradient Boosting | 75.21% | 75.18% |
| XGBoost | 73.61% | 74.01% |

**Finding:** Differences are negligible (<0.5%). Tree-based models learn optimal split thresholds automatically — manual clinical binning is unnecessary.

---

## Key Conclusions

- Model performance **plateaued at ~73–78%** across all strong non-linear classifiers — reflecting feature-set limitations, not model choice
- The dataset lacks biomarkers, genetic indicators, and longitudinal history — richer data is needed for further accuracy gains, not more complex algorithms
- **Deploy Random Forest at threshold 0.30–0.35 for screening**, escalating flagged patients to diagnostic workup

---

## Repository Structure

```
cardiovascular-disease-prediction/
│
├── health_data.csv                                  # Raw dataset
├── Cardiovascular_Disease_Prediction.ipynb          # Full modelling notebook
├── Cardiovascular_Disease_Prediction_FINAL_Report.pdf  # Project report
└── README.md
```

---

## Tools & Libraries

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Seaborn` · `Matplotlib`
