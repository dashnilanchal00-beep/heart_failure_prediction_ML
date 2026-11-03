# Heart Failure Death Prediction (Machine Learning)

A machine learning project that compares five classification algorithms for predicting the `DEATH_EVENT` outcome from the Heart Failure Clinical Records dataset.

> **Project type:** Learning / experimentation project  
> **Important:** This project is for educational purposes only. It is not a medical device, clinical decision-support system, or substitute for advice from qualified healthcare professionals.

## Overview

The goal is to compare several classifiers on a binary classification task:

- `DEATH_EVENT = 0`: the patient did not experience the recorded death event.
- `DEATH_EVENT = 1`: the patient experienced the recorded death event.

The experiment focuses on three evaluation metrics: accuracy, ROC-AUC, and recall for the positive class (`DEATH_EVENT = 1`). Recall is important to examine because a model can achieve reasonable accuracy while failing to identify a substantial share of positive cases.

## Repository contents

```text
heart_failure_prediction_ML/
├── README.md
├── code.py
└── heart_failure_clinical_records_dataset.csv
```

The dataset CSV is included in this repository and is read directly by `code.py` from the project folder.

## Dataset

The file `heart_failure_clinical_records_dataset.csv` contains **299 records and 13 columns**. The target column is `DEATH_EVENT`.

The script removes `time` from the predictor set because it represents follow-up duration and may contain information that would not be available at the intended prediction point. After excluding `time` and the target, the models use **11 predictor features**:

- Numeric features: `age`, `creatinine_phosphokinase`, `ejection_fraction`, `platelets`, `serum_creatinine`, and `serum_sodium`.
- Binary/categorical features: `anaemia`, `diabetes`, `high_blood_pressure`, `sex`, and `smoking`.

The target is imbalanced: approximately 68% of records have `DEATH_EVENT = 0` and 32% have `DEATH_EVENT = 1`.

## Methodology

### 1. Initial data checks

The script prints a preview of the dataset, column names, descriptive statistics, data types, missing-value counts, duplicate-row count, zero counts, and target-class counts.

### 2. Feature preparation

- `time` and `DEATH_EVENT` are excluded from the input features.
- The five binary/categorical columns are converted to the pandas `category` dtype.
- Numeric columns are standardized with `StandardScaler`.
- Categorical columns are one-hot encoded with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

### 3. Train/test splitting and preprocessing

The script runs **20 stratified train/test splits**. Each split uses 80% of the records for training and 20% for testing. The random-state values are `0, 5, 10, ..., 95`.

For each split, the scaler and encoder are fitted only on the training data. The fitted transformations are then applied to the corresponding test data. This avoids fitting preprocessing steps on the test set.

### 4. Models compared

The script compares these five classifiers using their current scikit-learn default settings, except where noted:

| Model | Configuration in `code.py` |
|---|---|
| K-Nearest Neighbors (KNN) | `KNeighborsClassifier()`; scikit-learn default is 5 neighbors |
| Logistic Regression | `LogisticRegression(max_iter=10000)` |
| Gaussian Naive Bayes | `GaussianNB()` |
| Linear Support Vector Machine | `SVC(kernel="linear", probability=True)` |
| RBF Support Vector Machine | `SVC(kernel="rbf", probability=True)` |

A majority-class baseline is also calculated. It always predicts `DEATH_EVENT = 0` (no recorded death event).

### 5. Evaluation metrics

Metrics are calculated on each test split and then averaged across the 20 splits:

- **Accuracy:** proportion of test predictions that are correct.
- **ROC-AUC:** calculated from the model's estimated probability for `DEATH_EVENT = 1`.
- **Positive-class recall:** proportion of actual `DEATH_EVENT = 1` records correctly identified by the model.

The baseline has no ROC-AUC value in the script and has zero recall for the positive class because it always predicts class 0.

## Results

The following are the mean values currently documented for the 20 splits in this project:

| Model | Mean Accuracy | Mean ROC-AUC | Mean Recall (`DEATH_EVENT = 1`) |
|---|---:|---:|---:|
| Linear SVM | 0.751 | 0.768 | 0.476 |
| RBF SVM | 0.739 | 0.775 | 0.479 |
| Logistic Regression | 0.737 | 0.766 | 0.461 |
| KNN (5 neighbors) | 0.700 | 0.687 | 0.287 |
| Gaussian Naive Bayes | 0.699 | 0.731 | 0.287 |
| Majority-class baseline | 0.683 | — | 0.000 |

### Key observations

- Linear SVM has the highest recorded mean accuracy (0.751).
- RBF SVM has the highest recorded mean ROC-AUC (0.775) and positive-class recall (0.479).
- The top three models have fairly close results; no single model performs best on every metric.
- Positive-class recall is below 0.50 for all five classifiers in the reported results. This means the models miss more than half of the positive cases at the classifiers' default prediction threshold.
- Accuracy should not be considered on its own, particularly because the target classes are imbalanced.

These scores are experimental results on repeated holdout splits of this dataset. They should not be interpreted as evidence of clinical performance or expected performance on data from another hospital or population.

## How to run the project

### Requirements

- Python
- NumPy
- pandas
- scikit-learn 1.2 or newer

The scikit-learn version requirement is needed because the code uses the `sparse_output` argument in `OneHotEncoder`.

### 1. Clone the repository

```bash
git clone https://github.com/dashnilanchal00-beep/heart_failure_prediction_ML.git
cd heart_failure_prediction_ML
```

### 2. Install the dependencies

```bash
python -m pip install numpy pandas "scikit-learn>=1.2"
```

### 3. Run the script

```bash
python code.py
```

Run the command from the project directory so that the script can find `heart_failure_clinical_records_dataset.csv`.

### Expected output

The script prints:

- Dataset preview, column information, descriptive statistics, and data checks.
- A model comparison table with average accuracy and ROC-AUC.
- A recall comparison table for the positive class.
- Counts of metric wins/ties across the repeated splits.
- The average accuracy of the majority-class baseline.

The current script prints results to the terminal. It does not save a trained model, prediction file, chart, or report to disk.

## Limitations

- The dataset is small (299 records) and comes from a single source.
- The 20 train/test splits reuse many of the same records, so the 20 results are not independent evaluations.
- The models use default hyperparameters; systematic hyperparameter tuning is not performed.
- The reported recall values for `DEATH_EVENT = 1` are low at the default classification threshold.
- The script does not currently report precision, F1-score, specificity, or confusion matrices.
- There is no external validation on data from other hospitals or populations.
- The script checks for missing values but does not implement missing-value imputation.

## Possible future improvements

- Add precision, F1-score, specificity, and confusion-matrix reporting.
- Examine the precision–recall trade-off at different probability thresholds, using validation data rather than selecting a threshold on the test set.
- Tune hyperparameters using cross-validation within the training data.
- Evaluate additional models, such as Random Forest and Gradient Boosting.
- Investigate transformations for highly skewed numeric features.
- Validate the approach on an independent dataset before making any claims about generalisation.

## Technology stack

- Python
- NumPy
- pandas
- scikit-learn

## Author

**Nilanchal Dash**  
B.Tech CSE (AI/ML)

## License and data attribution

A license file is not currently present in this repository. Do not assume that the code or dataset is licensed for unrestricted reuse. Before redistributing the dataset or reusing third-party material, check and add the applicable source attribution and license information.
