Heart Failure Death Prediction Using Machine Learning

Project Overview

This project focuses on predicting the risk of death among patients with heart failure using machine learning classification algorithms.

The project uses the Heart Failure Clinical Records Dataset to analyze clinical features and compare the performance of multiple machine learning models in predicting the death event outcome.

The main objective is to explore how machine learning techniques can be applied to healthcare-related data and evaluate different classification algorithms using appropriate performance metrics.

Objectives

- Analyze clinical records of patients with heart failure.
- Perform data cleaning and preprocessing.
- Identify relevant features for prediction.
- Train and evaluate multiple machine learning classification models.
- Compare model performance using different evaluation metrics.
- Identify the model with the most suitable predictive performance.

Dataset Information

Dataset: Heart Failure Clinical Records Dataset

The dataset contains clinical information about patients with heart failure.

It includes the following features:

Feature| Description
age| Age of the patient
anaemia| Whether the patient has anaemia
creatinine_phosphokinase| Level of CPK enzyme in the blood
diabetes| Whether the patient has diabetes
ejection_fraction| Percentage of blood leaving the heart at each contraction
high_blood_pressure| Whether the patient has high blood pressure
platelets| Platelet count in the blood
serum_creatinine| Level of serum creatinine in the blood
serum_sodium| Level of serum sodium in the blood
sex| Sex of the patient as encoded in the dataset
smoking| Whether the patient smokes
time| Follow-up period in days
DEATH_EVENT| Target variable indicating whether the patient experienced the recorded death event

Target Variable: "DEATH_EVENT"

- "0" – No recorded death event
- "1" – Recorded death event

The "time" feature is excluded during model training, leaving 11 predictor features.

The CSV dataset is included in this repository.

Machine Learning Algorithms Used

Five classification algorithms are implemented and compared:

1. K-Nearest Neighbors (KNN) – Classifies observations based on their nearest neighbors.
2. Logistic Regression – Uses a logistic function for binary classification.
3. Gaussian Naive Bayes – Applies Bayes' theorem with a Gaussian feature distribution assumption.
4. Linear Support Vector Machine (SVM) – Finds a linear decision boundary between classes.
5. RBF Support Vector Machine (SVM) – Uses a radial basis function kernel to model nonlinear decision boundaries.

Technologies and Libraries

The project uses the following technologies:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Project Structure

heart_failure_prediction_ML/
│
├── heart_failure_clinical_records_dataset.csv
├── code.py
└── README.md

Data Preprocessing

The following preprocessing steps are performed:

- Loading the dataset using Pandas.
- Checking for missing values.
- Checking for duplicate records.
- Separating input features and the target variable.
- Removing the "time" feature from the predictors.
- Identifying numerical and categorical features.
- Applying StandardScaler to numerical features.
- Applying OneHotEncoder to categorical features.
- Splitting the dataset into training and testing sets using stratified sampling.

Preprocessing is fitted using training data and then applied to test data to help prevent data leakage.

Model Evaluation

The models are evaluated using the following metrics:

- Accuracy: Measures the overall proportion of correct predictions.
- ROC-AUC Score: Measures the model's ability to distinguish between the two classes.
- Recall: Measures how many actual positive cases are correctly identified.

The models are evaluated over 20 stratified splits to compare their average performance.

Results and Model Comparison

According to the experimental results documented for this project:

Model| Average Accuracy
Linear SVM| 75.1%
RBF SVM| 73.9%
Logistic Regression| 73.7%
K-Nearest Neighbors| 70.0%
Gaussian Naive Bayes| 69.9%

Key Findings

- Linear SVM achieved the highest average accuracy of 75.1%.
- RBF SVM achieved the highest ROC-AUC and death-class recall among the evaluated models.
- Logistic Regression also demonstrated competitive classification performance.
- Accuracy alone does not provide a complete picture of model performance, particularly when evaluating healthcare-related datasets.

Best Model: Linear SVM based on average accuracy.

However, RBF SVM may be preferable when considering ROC-AUC and recall.

Installation and Setup

1. Clone the Repository

git clone https://github.com/dashnilanchal00-beep/heart_failure_prediction_ML.git

2. Navigate to the Project Directory

cd heart_failure_prediction_ML

3. Install Required Libraries

pip install numpy pandas matplotlib seaborn scikit-learn

Note: The code uses "OneHotEncoder(sparse_output=False)", which requires scikit-learn version 1.2 or newer.

4. Run the Project

python code.py

Limitations

- The dataset contains a relatively small number of patient records.
- The dataset may not represent all patient populations.
- Model performance depends on the quality and characteristics of the available data.
- Predictive performance on this dataset does not guarantee performance on unseen real-world clinical populations.
- The models are developed for educational and experimental purposes only.

Future Improvements

Some possible improvements include:

- Experimenting with additional machine learning algorithms.
- Applying hyperparameter tuning.
- Performing cross-validation.
- Adding confusion matrices and classification reports.
- Comparing precision, recall, F1-score, and ROC-AUC.
- Exploring feature importance and model interpretability.
- Testing models on additional independent datasets.

Conclusion

This project demonstrates the application of machine learning classification techniques to heart failure clinical records.

By comparing five different classification algorithms, the project explores how different models perform in predicting the recorded death event.

The results show that Linear SVM achieved the highest average accuracy, while RBF SVM performed better in terms of ROC-AUC and death-class recall.

The project highlights the importance of evaluating machine learning models using multiple performance metrics rather than relying on accuracy alone.

Disclaimer

This project is developed strictly for educational and research purposes. It is not a clinically validated medical prediction system and must not be used for medical diagnosis, treatment decisions, or patient care.

Author

Dashni Lanchal

GitHub: https://github.com/dashnilanchal00-beep

---

⭐ If you find this project interesting, consider giving the repository a star!
