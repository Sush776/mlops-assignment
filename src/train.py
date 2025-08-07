# src/train.py

import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Set up MLflow
mlflow.set_experiment("iris_classification")

def train_and_log_model(model, model_name, params: dict):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ {model_name} accuracy: {acc:.4f}")

if __name__ == "__main__":
    # Logistic Regression
    lr = LogisticRegression(max_iter=200)
    train_and_log_model(lr, "LogisticRegression", {"max_iter": 200})

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=3)
    train_and_log_model(rf, "RandomForest", {"n_estimators": 100, "max_depth": 3})
