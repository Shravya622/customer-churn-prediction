import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"

MODEL_PATH = BASE_DIR / "models" / "best_churn_model.pkl"

THRESHOLD_PATH = BASE_DIR / "models" / "model_threshold.txt"

# Create MLflow experiment
mlflow.set_experiment("Customer Churn Prediction")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove customer ID
df = df.drop(columns=["customerID"])

# Features and target
X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

    return metrics

    # ==============================
# Logistic Regression
# ==============================

with mlflow.start_run(run_name="Logistic Regression"):

    # Create pipeline
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]
    )

    # Train model
    logistic_pipeline.fit(X_train, y_train)

    # Predictions
    y_pred_lr = logistic_pipeline.predict(X_test)
    y_prob_lr = logistic_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate
    lr_metrics = evaluate_model(
        y_test,
        y_pred_lr,
        y_prob_lr
    )

    # Log parameters
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("accuracy", lr_metrics["accuracy"])
    mlflow.log_metric("precision", lr_metrics["precision"])
    mlflow.log_metric("recall", lr_metrics["recall"])
    mlflow.log_metric("f1_score", lr_metrics["f1_score"])
    mlflow.log_metric("roc_auc", lr_metrics["roc_auc"])

    # Save model in MLflow
    mlflow.sklearn.log_model(
        logistic_pipeline,
        artifact_path="model"
    )

print("Logistic Regression logged successfully.")

# ==============================
# Random Forest
# ==============================

with mlflow.start_run(run_name="Random Forest"):

    # Create pipeline
    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    # Train model
    random_forest_pipeline.fit(X_train, y_train)

    # Predictions
    y_pred_rf = random_forest_pipeline.predict(X_test)
    y_prob_rf = random_forest_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate
    rf_metrics = evaluate_model(
        y_test,
        y_pred_rf,
        y_prob_rf
    )

    # Log parameters
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("accuracy", rf_metrics["accuracy"])
    mlflow.log_metric("precision", rf_metrics["precision"])
    mlflow.log_metric("recall", rf_metrics["recall"])
    mlflow.log_metric("f1_score", rf_metrics["f1_score"])
    mlflow.log_metric("roc_auc", rf_metrics["roc_auc"])

    # Save model
    mlflow.sklearn.log_model(
        random_forest_pipeline,
        artifact_path="model"
    )

print("Random Forest logged successfully.")

# ==============================
# XGBoost
# ==============================

with mlflow.start_run(run_name="XGBoost"):

    # Create pipeline
    xgboost_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    # Train model
    xgboost_pipeline.fit(X_train, y_train)

    # Predictions
    y_pred_xgb = xgboost_pipeline.predict(X_test)
    y_prob_xgb = xgboost_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate
    xgb_metrics = evaluate_model(
        y_test,
        y_pred_xgb,
        y_prob_xgb
    )

    # Log parameters
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("accuracy", xgb_metrics["accuracy"])
    mlflow.log_metric("precision", xgb_metrics["precision"])
    mlflow.log_metric("recall", xgb_metrics["recall"])
    mlflow.log_metric("f1_score", xgb_metrics["f1_score"])
    mlflow.log_metric("roc_auc", xgb_metrics["roc_auc"])

    # Save model
    mlflow.sklearn.log_model(
        xgboost_pipeline,
        artifact_path="model"
    )

print("XGBoost logged successfully.")

# ==============================
# Save Final Model
# ==============================

joblib.dump(logistic_pipeline, MODEL_PATH)

with open(THRESHOLD_PATH, "w") as file:
    file.write("0.40")

print("\nFinal model saved successfully.")
print("Selected Model : Logistic Regression")
print("Threshold      : 0.40")

