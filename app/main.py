from pathlib import Path

import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from app.s3_utils import download_model_from_s3


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "registered_model"

# Download the registered model from S3 if it is not available locally.
if not MODEL_DIR.exists():
    download_model_from_s3(MODEL_DIR)

# Load the MLflow model.
model = mlflow.sklearn.load_model(str(MODEL_DIR))

CHURN_THRESHOLD = 0.40


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn risk using a trained Logistic Regression pipeline.",
    version="1.0.0"
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Logistic Regression",
        "model_source": "MLflow model from S3",
        "churn_threshold": CHURN_THRESHOLD
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    customer_df = pd.DataFrame([customer.model_dump()])

    churn_probability = float(model.predict_proba(customer_df)[:, 1][0])
    prediction = "Churn" if churn_probability >= CHURN_THRESHOLD else "No Churn"

    return {
        "prediction": prediction,
        "churn_probability": round(churn_probability, 4),
        "threshold_used": CHURN_THRESHOLD
    }