# 🫀 Cardiovascular Disease Prediction using Machine Learning

## 📌 Project Overview
This project develops and evaluates multiple machine learning models to predict the presence of cardiovascular disease using a structured medical dataset containing **70,000 patient records**.

The objective was not only to build a predictive model, but to:
* **Systematically compare** multiple algorithms.
* **Control overfitting** using cross-validation.
* **Analyze** bias–variance trade-offs.
* **Interpret** feature importance.
* **Perform threshold optimization** based on clinical context.

---

## 📅 Project Timeline
* **Started:** 12 February
* **Completed:** 19 February
* *(Note: Extended experimentation was required due to large model training times, particularly for SVM and ensemble tuning.)*

---

## 📊 Dataset Description
The dataset includes the following patient features:
* **Demographics:** Age, Gender, Height & Weight (engineered into BMI).
* **Clinical Metrics:** Systolic Blood Pressure (`ap_hi`), Diastolic Blood Pressure (`ap_lo`), Cholesterol, Glucose.
* **Lifestyle:** Smoking, Alcohol Intake, Physical Activity.

**🎯 Target Variable:**
* `0` → No Disease
* `1` → Cardiovascular Disease

> [!NOTE]
> The dataset is approximately balanced (~50% positive, ~50% negative cases).

---

## 🔎 Data Preprocessing & Feature Engineering
Medical domain reasoning was applied during data cleaning instead of relying solely on statistical outlier detection:

1.  **Cleaning:** Removed unnecessary columns (`ID`, `unnamed`) and minimal missing values (7 rows).
2.  **Conversion:** Converted age from days to years.
3.  **Engineering:** Calculated **BMI** from height and weight.
4.  **Blood Pressure Correction:**
    * Removed impossible values (e.g., zero BP).
    * Fixed typographical errors (extra zeros).
    * Removed physiologically unrealistic values (>300 mmHg).
5.  **Validation:** Verified BMI range (10–100) and gender encoding through mean height comparison.

---

## 🧠 Modeling Strategy
### Train–Test Strategy
* **Stratified train–test split** to maintain class proportions.
* **5-fold cross-validation** for fair performance comparison.
* **Out-of-Bag (OOB) validation** specifically for Random Forest.

### Models Evaluated
* Logistic Regression & Polynomial Logistic Regression
* Decision Tree & Random Forest
* Gradient Boosting & XGBoost
* Support Vector Machine (SVM) & PCA + SVM
* K-Nearest Neighbors
* Bagging & Voting Classifiers

---

## 📈 Model Comparison Summary

| Model | Cross-Validated Performance |
| :--- | :--- |
| **Random Forest** | **~75–76%** |
| Gradient Boosting | ~75% |
| XGBoost | ~74% |
| Decision Tree | ~73% |
| Logistic Regression | ~72% |
| SVM (CV) | ~72% |

**Observation:** Performance plateaued across strong nonlinear models, indicating a feature-set predictive limitation or the presence of irreducible noise. No significant improvement was found by increasing model complexity beyond Random Forest.

---

## 🏆 Final Model Selection
**Final Model:** Random Forest (Original Continuous Dataset)

**Selected due to:**
* Stable cross-validation performance (~75–76%).
* OOB score consistency (~73%).
* Controlled variance and computational efficiency.
* Clear interpretability via feature importance.

---

## 🔬 Feature Importance Insights
The top predictors identified by the model are:
1.  **Systolic Blood Pressure (~50% importance)**
2.  **Diastolic Blood Pressure**
3.  **Age**
4.  **Cholesterol**

Lifestyle-related variables (smoking, alcohol, activity) showed a comparatively lower direct predictive impact in this specific dataset.

---

## 🎯 Threshold Optimization
The default threshold (0.5) produced a Precision of 0.75 and a Recall of 0.69. To reduce false negatives (critical in medical screening), threshold tuning was performed:

| Threshold | Precision | Recall |
| :--- | :--- | :--- |
| 0.30 | 62% | **89%** |
| 0.35 | 66% | 85% |
| 0.40 | 69% | 80% |
| **0.50 (Default)** | **75%** | **69%** |

* **Lower Thresholds:** Higher recall (Better for mass screening).
* **Higher Thresholds:** Higher precision (Better for diagnostic confirmation).

---

## 📌 Key Insights & Limitations
* **Complexity:** Increasing model complexity did not significantly improve performance beyond ~75–78%.
* **Drivers:** Blood pressure features dominate predictive power.
* **Cost Sensitivity:** Real-world deployment requires selecting thresholds based on clinical objective and the "cost" of missing a diagnosis.
* **Limitations:** The study is limited to basic clinical features (no biomarkers/imaging); feature importance in trees can be biased toward continuous variables.

---

## 🚀 Technical Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## 🧩 Learning Outcomes
This project strengthened my understanding of:
* Bias–variance tradeoff
* Bias–variance tradeoff
* Cross-validation discipline
* Model generalization vs training performance
* Threshold optimization
* Clinical interpretation of ML predictions
* Computational trade-offs (e.g., SVM training time constraints)

---



## 📬 Final Note
This project emphasizes structured experimentation and decision-aware modeling rather than simply maximizing accuracy. The goal was to approach the problem as an ML practitioner—focusing on generalization, interpretability, and real-world applicability.
