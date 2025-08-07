# src/data_preprocessing.py

import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_and_save_raw_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/iris.csv", index=False)
    print("✅ Raw data saved to data/raw/iris.csv")
    return df

def preprocess_and_split_data(df):
    os.makedirs("data/processed", exist_ok=True)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("✅ Processed data saved to data/processed/")

if __name__ == "__main__":
    df = load_and_save_raw_data()
    preprocess_and_split_data(df)
