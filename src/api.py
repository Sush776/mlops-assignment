# src/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import numpy as np

app = FastAPI()

# Load model
#model = mlflow.sklearn.load_model("models:/iris_best_model/1")
# Load model from local folder
model = mlflow.sklearn.load_model("models/iris_best_model")


# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Iris prediction API is running."}

@app.post("/predict")
def predict(features: IrisFeatures):
    data = pd.DataFrame([features.dict()])
    
    # Rename input columns to match training-time column names
    data.columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}


