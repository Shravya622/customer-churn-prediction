from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# Paths are built relative to this file, so the API works from the project folder.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_churn_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "model_threshold.txt"

# Load the trained pipeline and selected business threshold once when the API starts.
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    CHURN_THRESHOLD = float(file.read().strip())


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
        "churn_threshold": CHURN_THRESHOLD
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Convert the validated request into one-row DataFrame for the ML pipeline.
    customer_df = pd.DataFrame([customer.model_dump()])

    churn_probability = float(model.predict_proba(customer_df)[:, 1][0])
    prediction = "Churn" if churn_probability >= CHURN_THRESHOLD else "No Churn"

    return {
        "prediction": prediction,
        "churn_probability": round(churn_probability, 4),
        "threshold_used": CHURN_THRESHOLD
    }