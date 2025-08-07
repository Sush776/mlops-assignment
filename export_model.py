# export_model.py

import mlflow.sklearn

# Load model from registry (name + version)
model = mlflow.sklearn.load_model("models:/iris_best_model/1")

# Save to local folder
mlflow.sklearn.save_model(model, path="models/iris_best_model")

print("✅ Model exported to models/iris_best_model/")
