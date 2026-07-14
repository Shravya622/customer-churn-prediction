# Customer Churn Prediction with MLOps and AWS Deployment

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn. The project covers data preprocessing, model training and evaluation, experiment tracking with MLflow, threshold tuning, REST API development using FastAPI, containerization with Docker, model storage in Amazon S3, and cloud deployment using AWS Elastic Beanstalk.

## Project Overview

Customer churn is an important business problem in the telecommunications industry. Identifying customers who are likely to leave allows businesses to take proactive retention measures.

This project builds and deploys a machine learning pipeline that accepts customer information and returns:

- Churn prediction: `Churn` or `No Churn`
- Churn probability
- Classification threshold used for prediction

The final model is exposed through a publicly accessible REST API with interactive Swagger documentation.

## Project Workflow

```text
Telco Customer Churn Dataset
          |
          v
Data Cleaning and Preprocessing
          |
          v
Exploratory Data Analysis
          |
          v
Model Training and Evaluation
          |
          +-------------------+
          |                   |
          v                   v
 Logistic Regression    Random Forest / XGBoost
          |
          v
Threshold Tuning
          |
          v
MLflow Experiment Tracking
          |
          v
Final Model Pipeline
          |
          v
Amazon S3 Model Storage
          |
          v
FastAPI Prediction API
          |
          v
Docker Container
          |
          v
AWS Elastic Beanstalk
          |
          v
Public Prediction API
```

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

- Demographic information
- Account tenure
- Internet service
- Phone service
- Contract type
- Payment method
- Monthly charges
- Total charges
- Customer churn status

The `customerID` column was removed because it does not provide useful predictive information.

Missing values in `TotalCharges` were handled during preprocessing.

## Data Preprocessing

The preprocessing pipeline includes:

- Removal of unnecessary identifiers
- Handling of missing values
- Standardization of numerical features using `StandardScaler`
- Encoding of categorical features using `OneHotEncoder`
- Train-test splitting
- Integration of preprocessing and prediction into a reusable machine learning pipeline

Dataset dimensions:

```text
Feature matrix: (7043, 19)
Target vector:  (7043,)
```

After preprocessing:

```text
Training data: (5634, 45)
Testing data:  (1409, 45)
```

Training target distribution:

```text
No Churn: 73.46%
Churn:    26.54%
```

## Models Evaluated

Three machine learning models were trained and compared.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8421 |
| Random Forest | 0.7835 | 0.6211 | 0.4733 | 0.5372 | 0.8227 |
| XGBoost | 0.7970 | 0.6438 | 0.5267 | 0.5794 | 0.8411 |

Logistic Regression was selected as the final model based on its overall performance and ROC-AUC score.

## Threshold Tuning

The default classification threshold of `0.50` was adjusted to `0.40` to improve the model's ability to identify customers at risk of churn.

Final Logistic Regression results at threshold `0.40`:

| Metric | Score |
|---|---:|
| Accuracy | 0.7771 |
| Precision | 0.5682 |
| Recall | 0.6684 |
| F1-Score | 0.6143 |

The deployed API therefore uses:

```text
Classification Threshold = 0.40
```

## MLflow Experiment Tracking

MLflow was used to track machine learning experiments.

The MLflow experiment was named:

```text
Customer Churn Prediction
```

Tracked information includes:

- Model parameters
- Evaluation metrics
- Model comparison
- Experiment runs
- Model artifacts

The experiments included:

- Logistic Regression
- Random Forest
- XGBoost

The final selected model was Logistic Regression with a classification threshold of `0.40`.

## API Development

The prediction service is built using FastAPI.

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API home endpoint |
| GET | `/health` | Health check |
| POST | `/predict` | Predict customer churn |
| GET | `/docs` | Interactive Swagger API documentation |

## Prediction Request Example

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}
```

## Prediction Response Example

```json
{
  "prediction": "Churn",
  "churn_probability": 0.7138,
  "threshold_used": 0.4
}
```

## Model Storage with Amazon S3

The trained model artifact is stored in Amazon S3.

During application startup, the deployed application retrieves the required model artifact from S3 and loads it for inference.

This separates the model artifact from the application source code and avoids packaging the trained model directly inside the deployment bundle.

## Docker Containerization

The FastAPI application is containerized using Docker.

The Docker image contains:

- Python runtime
- Application source code
- Required Python dependencies
- FastAPI application server configuration

Docker provides a consistent runtime environment for local execution and cloud deployment.

## AWS Deployment

The application is deployed using AWS Elastic Beanstalk on a Docker platform.

Deployment architecture:

```text
Client
   |
   v
AWS Elastic Beanstalk Public URL
   |
   v
Nginx Reverse Proxy
   |
   v
Docker Container
   |
   v
FastAPI Application
   |
   v
Model Artifact from Amazon S3
   |
   v
Logistic Regression Pipeline
   |
   v
Churn Prediction Response
```

AWS services used:

- AWS Elastic Beanstalk
- Amazon EC2
- Amazon S3
- AWS IAM

## Live API

The deployed application can be accessed through the following endpoints:

### API Base URL

`http://customer-churn-api-env.eba-sw3ds35u.eu-north-1.elasticbeanstalk.com`

### Swagger API Documentation

`http://customer-churn-api-env.eba-sw3ds35u.eu-north-1.elasticbeanstalk.com/docs`

The Swagger interface can be used to test the `/predict` endpoint directly from a web browser.

> Note: The public endpoint remains available while the AWS Elastic Beanstalk environment is running.

## Project Structure

```text
customer-churn-prediction/
|
|-- app/
|   |-- main.py
|   `-- s3_utils.py
|
|-- data/
|-- notebooks/
|-- src/
|
|-- .dockerignore
|-- .gitignore
|-- config.py
|-- Dockerfile
|-- README.md
|-- requirements.txt
`-- upload_to_s3.py
```

Local virtual environments, environment variables, MLflow local data, generated model artifacts, and deployment ZIP files are excluded from version control.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- MLflow
- FastAPI
- Docker
- Amazon S3
- AWS Elastic Beanstalk
- AWS IAM
- Git
- GitHub

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shravya622/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure required environment variables

Create a local `.env` file if required by the application configuration.

Do not commit `.env` files or AWS credentials to GitHub.

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Key Features

- End-to-end machine learning workflow
- Multiple model comparison
- Classification threshold tuning
- MLflow experiment tracking
- Reusable preprocessing and prediction pipeline
- REST API using FastAPI
- Interactive Swagger documentation
- Docker containerization
- Model artifact storage in Amazon S3
- AWS IAM-based access control
- Cloud deployment using AWS Elastic Beanstalk
- Publicly accessible prediction endpoint

## Author

**Shravya G Naik**

Computer Science and Engineering (Data Science)