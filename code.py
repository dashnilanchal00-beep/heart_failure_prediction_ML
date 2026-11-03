import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.svm import SVC

warnings.filterwarnings('ignore', category=FutureWarning)

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

pd.set_option('display.max_columns', None)

print(df.head())
print(df.columns)
print(df.describe())

print(df.isna().sum())
print(df.duplicated().sum())

print((df == 0).sum())

print(df['DEATH_EVENT'].value_counts())
print(df.columns)

categorical_columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

for col in categorical_columns:
    df[col] = df[col].astype('category')

print(df.dtypes)

X = df.drop(columns=['time', 'DEATH_EVENT'])
y = df['DEATH_EVENT']

knn_acc = []
knn_auc = []
knn_recall = []

log_acc = []
log_auc = []
log_recall = []

nb_acc = []
nb_auc = []
nb_recall = []

lin_acc = []
lin_auc = []
lin_recall = []

rbf_acc = []
rbf_auc = []
rbf_recall = []

baseline_acc = []

for value in range(0, 100, 5):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=value,
        stratify=y
    )

    # Baseline model: always predicts no death (DEATH_EVENT = 0)
    baseline_accuracy = 1 - y_test.mean()
    baseline_acc.append(baseline_accuracy)

    X_train_num = X_train.select_dtypes(include='number')
    X_train_cla = X_train.select_dtypes(include='category')
    X_test_num = X_test.select_dtypes(include='number')
    X_test_cla = X_test.select_dtypes(include='category')

    scaler = StandardScaler()
    X_train_num_sc = scaler.fit_transform(X_train_num)
    X_test_num_sc = scaler.transform(X_test_num)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cla_en = encoder.fit_transform(X_train_cla)
    X_test_cla_en = encoder.transform(X_test_cla)

    X_train_final = np.hstack((X_train_num_sc, X_train_cla_en))
    X_test_final = np.hstack((X_test_num_sc, X_test_cla_en))

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train_final, y_train)

    knn_accuracy = knn.score(X_test_final, y_test)
    knn_predict = knn.predict(X_test_final)
    knn_predict_proba = knn.predict_proba(X_test_final)

    knn_auc_score = roc_auc_score(y_test, knn_predict_proba[:, 1])
    knn_recall_score = recall_score(y_test, knn_predict, pos_label=1, zero_division=0)

    knn_acc.append(knn_accuracy)
    knn_auc.append(knn_auc_score)
    knn_recall.append(knn_recall_score)

    # Logistic Regression
    log = LogisticRegression(max_iter=10000)
    log.fit(X_train_final, y_train)

    log_accuracy = log.score(X_test_final, y_test)
    log_predict = log.predict(X_test_final)
    log_predict_proba = log.predict_proba(X_test_final)

    log_auc_score = roc_auc_score(y_test, log_predict_proba[:, 1])
    log_recall_score = recall_score(y_test, log_predict, pos_label=1, zero_division=0)

    log_acc.append(log_accuracy)
    log_auc.append(log_auc_score)
    log_recall.append(log_recall_score)

    # Gaussian Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_final, y_train)

    nb_accuracy = nb.score(X_test_final, y_test)
    nb_predict = nb.predict(X_test_final)
    nb_predict_proba = nb.predict_proba(X_test_final)

    nb_auc_score = roc_auc_score(y_test, nb_predict_proba[:, 1])
    nb_recall_score = recall_score(y_test, nb_predict, pos_label=1, zero_division=0)

    nb_acc.append(nb_accuracy)
    nb_auc.append(nb_auc_score)
    nb_recall.append(nb_recall_score)

    # Linear SVM
    lin = SVC(kernel='linear', probability=True)
    lin.fit(X_train_final, y_train)

    lin_accuracy = lin.score(X_test_final, y_test)
    lin_predict = lin.predict(X_test_final)
    lin_predict_proba = lin.predict_proba(X_test_final)

    lin_auc_score = roc_auc_score(y_test, lin_predict_proba[:, 1])
    lin_recall_score = recall_score(y_test, lin_predict, pos_label=1, zero_division=0)

    lin_acc.append(lin_accuracy)
    lin_auc.append(lin_auc_score)
    lin_recall.append(lin_recall_score)

    # RBF SVM
    rbf = SVC(kernel='rbf', probability=True)
    rbf.fit(X_train_final, y_train)

    rbf_accuracy = rbf.score(X_test_final, y_test)
    rbf_predict = rbf.predict(X_test_final)
    rbf_predict_proba = rbf.predict_proba(X_test_final)

    rbf_auc_score = roc_auc_score(y_test, rbf_predict_proba[:, 1])
    rbf_recall_score = recall_score(y_test, rbf_predict, pos_label=1, zero_division=0)

    rbf_acc.append(rbf_accuracy)
    rbf_auc.append(rbf_auc_score)
    rbf_recall.append(rbf_recall_score)


# ACCURACY RESULTS
accuracy_results = {
    'KNN': knn_acc,
    'Logistic Regression': log_acc,
    'Gaussian Naive Bayes': nb_acc,
    'Linear SVM': lin_acc,
    'RBF SVM': rbf_acc
}

# AUC RESULTS
auc_results = {
    'KNN': knn_auc,
    'Logistic Regression': log_auc,
    'Gaussian Naive Bayes': nb_auc,
    'Linear SVM': lin_auc,
    'RBF SVM': rbf_auc
}

# RECALL RESULTS
recall_results = {
    'KNN': knn_recall,
    'Logistic Regression': log_recall,
    'Gaussian Naive Bayes': nb_recall,
    'Linear SVM': lin_recall,
    'RBF SVM': rbf_recall
}

accuracy_df = pd.DataFrame(accuracy_results)
auc_df = pd.DataFrame(auc_results)
recall_df = pd.DataFrame(recall_results)


def calculate_wins(results, atol=1e-12):

    max_values = results.max(axis=1).to_numpy()[:, None]

    winners = np.isclose(
        results.to_numpy(),
        max_values,
        rtol=0,
        atol=atol
    )

    shared_wins = pd.Series(
        winners.sum(axis=0),
        index=results.columns
    )

    fractional_wins = pd.Series(
        (winners / winners.sum(axis=1, keepdims=True)).sum(axis=0),
        index=results.columns
    )

    tie_count = (winners.sum(axis=1) > 1).sum()

    return shared_wins, fractional_wins, tie_count


# ACCURACY WINS
accuracy_wins, accuracy_fractional, accuracy_ties = calculate_wins(accuracy_df)

# AUC WINS
auc_wins, auc_fractional, auc_ties = calculate_wins(auc_df)

# RECALL WINS
recall_wins, recall_fractional, recall_ties = calculate_wins(recall_df)


# ORIGINAL MODEL COMPARISON TABLE
comparison = pd.DataFrame({
    'Average Accuracy': accuracy_df.mean(),
    'Accuracy Wins': accuracy_wins,
    'Average AUC': auc_df.mean(),
    'AUC Wins': auc_wins,
    'Accuracy Fractional Wins': accuracy_fractional,
    'AUC Fractional Wins': auc_fractional
})


print("\nMODEL COMPARISON")

print(comparison.sort_values(
    by='Average Accuracy',
    ascending=False
).round(4))


# NEW RECALL COMPARISON TABLE
recall_comparison = pd.DataFrame({
    'Average Death Recall': recall_df.mean(),
    'Death Recall Wins': recall_wins,
    'Death Recall Fractional Wins': recall_fractional
})


print("\nRECALL COMPARISON")

print(recall_comparison.sort_values(
    by='Average Death Recall',
    ascending=False
).round(4))


# TIE COUNTS
print("\nAccuracy tie iterations:", accuracy_ties)
print("AUC tie iterations:", auc_ties)
print("Recall tie iterations:", recall_ties)


# BASELINE ACCURACY (PRINTED AT THE END)
print("\nBASELINE MODEL")
print("Average Baseline Accuracy:", round(np.mean(baseline_acc), 4))
