# Iris Flower Prediction API - MLOps Assignment

This project implements an end-to-end MLOps pipeline for the Iris flower classification problem. It includes model training, serving via FastAPI, monitoring with Prometheus, and retraining capabilities.

---

## Features

- Iris classification prediction API with FastAPI
- Input validation using Pydantic
- Logging predictions to CSV and SQLite
- Metrics endpoint (`/metrics`) for real-time usage stats
- Prometheus monitoring integration
- Retrain endpoint (`/retrain`) to update the model with new data
- Dockerized for easy deployment

---

## Setup & Run
docker build -t iris-api .
docker run -d -p 8000:8000 --name iris-api-container iris-api



API Endpoints
GET /
Health check endpoint. Returns a message confirming the API is running.

POST /predict
Takes JSON input with Iris features and returns the predicted class.
Example input:
Example 
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

To monitor the API, run Prometheus with the provided config:

docker run -d -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
Prometheus will scrape metrics from the API’s /metrics endpoint.

### Clone the repo

```bash
git clone https://github.com/Sush776/mlops-assignment.git
cd mlops-assignment
