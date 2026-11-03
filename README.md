# Heart Failure Death Prediction

Comparing classifiers for predicting in-hospital death (`DEATH_EVENT`) from clinical records, evaluated over 20 random stratified splits.

> **Status:** learning project (Phase 3 of my ML roadmap). Results are honest estimates, and the limitations are listed below.

## Problem
Given 12 clinical features of heart failure patients, predict whether the patient died during the follow-up period. Because a missed high-risk patient is costly, **recall on the "died" class** is treated as the key metric alongside accuracy and AUC.

## Dataset
- **Heart Failure Clinical Records** (Kaggle, `heart_failure_clinical_records_dataset.csv`)
- 299 patients, 13 columns. Target: `DEATH_EVENT` (0 = survived, 1 = died). About 32% died.
- The data file is not included in this repo. Download it from Kaggle and place it in the project folder.

## Approach

### 1. Data checks
- No missing values and no duplicate rows.
- Zeros in the binary columns (`anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`) are valid codes, not missing data.
- No impossible zeros in the physiological columns.

### 2. Feature decisions
- **Dropped `time`** (follow-up days). It is only known after the outcome, so it leaks the answer.
- Binary columns are one-hot encoded (`handle_unknown="ignore"`).
- Numeric columns are standardized (`StandardScaler`).
- Skewed columns (`creatinine_phosphokinase`, `serum_creatinine`, `platelets`) are kept and scaled, with no log transform yet.

### 3. Leakage-safe preprocessing
Split first (stratified), then fit the scaler and encoder on the training split only, and apply them to the test split.

### 4. Evaluation
- 20 stratified 80/20 splits (`random_state` = 0, 5, 10, ..., 95)
- Models: KNN (k=5), Logistic Regression, Gaussian Naive Bayes, Linear SVM, RBF SVM
- Metrics per split: accuracy, ROC-AUC (on P(died)), and recall for the "died" class
- A majority-class baseline (always predict "survived") is included for reference

## Results (mean over 20 splits)

| Model | Accuracy | AUC | Died recall |
|---|---|---|---|
| Linear SVM | 0.751 | 0.768 | 0.476 |
| RBF SVM | 0.739 | **0.775** | **0.479** |
| Logistic Regression | 0.737 | 0.766 | 0.461 |
| KNN (k=5) | 0.700 | 0.687 | 0.287 |
| Gaussian Naive Bayes | 0.699 | 0.731 | 0.287 |
| Baseline (always "survived") | 0.683 | n/a | 0.000 |

## Key Findings
- **Every model beats the baseline on accuracy**, but only by 2 to 7 points.
- **Accuracy hides a weakness.** The best models still miss about half of the patients who died (died recall around 0.48).
- **KNN and Naive Bayes** have the lowest died recall (about 0.29), so they are weaker for the costly error even though their accuracy is not far behind.
- **Winners depend on the metric.** Linear SVM leads accuracy, RBF SVM leads AUC and died recall. No single model wins everything.
- Differences among the top three models are within about 2 points, which is close to split-to-split noise.

## Limitations
- **Models were compared on the same test splits used for reporting, with default hyperparameters.** No tuning was done, so these numbers are a baseline, not a tuned best.
- **Died recall at the default 0.5 threshold is low.** A lower threshold on P(died) would raise recall at the cost of precision. This was not explored here.
- **Small dataset** (299 patients, about 60 in each test split), so each patient moves a score by about 1.7%.
- **The 20 splits share the same patients**, so they are not independent.
- **Retrospective data from a single source.** It won't generalise to other hospitals without validation.

## Possible Improvements
- Lower the threshold on P(died) and report the recall-precision trade-off
- Tune hyperparameters with cross-validation inside each training split
- Try tree-based models (Random Forest, Gradient Boosting)
- Log-transform the skewed columns

## How to Run
```bash
pip install numpy pandas scikit-learn
python p2_heart_failure.py
```
Place `heart_failure_clinical_records_dataset.csv` in the same folder as the script.

## Tech Stack
Python, NumPy, pandas, scikit-learn

## Author
Nilanchal Dash, B.Tech CSE (AI/ML)
