# src/api.py
import shutil

import csv
import sqlite3
from datetime import datetime
from collections import defaultdict
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = FastAPI()

# Load model
# model = mlflow.sklearn.load_model("models:/iris_best_model/1")
model = mlflow.sklearn.load_model("models/iris_best_model")

# Metrics dictionary
metrics = {
    "total_predictions": 0,
    "class_counts": defaultdict(int),
    "last_prediction_time": None
}

# Input schema
""" class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
 """


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


@app.get("/")
def read_root():
    return {"message": "Iris prediction API is running."}

@app.post("/predict")
def predict(input_data: IrisFeatures):
    data = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]]
    prediction = model.predict(data)[0]
    timestamp = datetime.now().isoformat()

    # Log to CSV file
    log_file = "prediction_logs.csv"
    log_data = [timestamp] + data[0] + [int(prediction)]
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_data)

    # Log to SQLite DB
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            prediction INTEGER
        )
    """)
    cursor.execute("""
        INSERT INTO predictions (timestamp, sepal_length, sepal_width, petal_length, petal_width, prediction)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, *data[0], int(prediction)))
    conn.commit()
    conn.close()

    # Update metrics
    metrics["total_predictions"] += 1
    metrics["class_counts"][int(prediction)] += 1
    metrics["last_prediction_time"] = timestamp

    return {"prediction": int(prediction)}

@app.get("/metrics")
def get_metrics():
    return {
        "total_predictions": metrics["total_predictions"],
        "class_counts": dict(metrics["class_counts"]),
        "last_prediction_time": metrics["last_prediction_time"]
    }



@app.post("/retrain")
def retrain_model():
    # Check if log file exists
    log_file = "prediction_logs.csv"
    if not os.path.exists(log_file):
        return {"error": "No prediction logs found for retraining."}

    # Load the CSV
    df = pd.read_csv(log_file, header=None)
    df.columns = [
        "timestamp",
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "prediction"
    ]

    # Features and target
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["prediction"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model (overwrite)
    model_path = "models/iris_best_model"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    mlflow.sklearn.save_model(clf, model_path)

    return {
        "message": "Model retrained and saved successfully.",
        "accuracy": accuracy,
        "samples_used": len(df)
    }
