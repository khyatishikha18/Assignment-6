# model_evaluation_and_tuning.py

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1. Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# 5. Evaluate each model
print("=== Model Evaluation ===")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))

# 6. Hyperparameter Tuning - Random Forest Example
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='f1_macro')
grid_rf.fit(X_train_scaled, y_train)
print("\nBest Parameters (GridSearchCV - RF):", grid_rf.best_params_)
best_rf = grid_rf.best_estimator_

# 7. RandomizedSearchCV - SVM Example
param_dist_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

random_svm = RandomizedSearchCV(SVC(), param_dist_svm, cv=5, n_iter=5, scoring='f1_macro', random_state=42)
random_svm.fit(X_train_scaled, y_train)
print("\nBest Parameters (RandomizedSearchCV - SVM):", random_svm.best_params_)
best_svm = random_svm.best_estimator_

# 8. Final Evaluation of Best Models
print("\n=== Final Evaluation of Tuned Models ===")
for name, model in [("Best RF", best_rf), ("Best SVM", best_svm)]:
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))
