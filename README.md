# Customer Churn Prediction API

## Overview

The Customer Churn Prediction API is a machine learning application that predicts whether a customer is likely to churn. The project uses a Random Forest model trained with Scikit-learn, tracks experiments using MLflow, stores model artifacts in AWS S3, and serves predictions through a FastAPI REST API.

---

## Features

* Customer churn prediction using Machine Learning
* FastAPI REST API
* MLflow model tracking and model registry
* AWS S3 integration for model storage
* Docker container support
* Health check endpoint
* JSON-based prediction API

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── app.py
├── train.py
├── config.py
├── download_from_s3.py
├── upload_to_s3.py
├── requirements.txt
├── Dockerfile
├── data/
├── registered_model/
├── models/
├── .gitignore
├── .dockerignore
└── README.md
```

---

## Technologies Used

* Python 3
* FastAPI
* Scikit-learn
* MLflow
* AWS S3
* Docker
* NumPy
* Pandas
* Boto3

---

## Machine Learning Workflow

1. Load customer churn dataset.
2. Clean missing values.
3. Encode categorical variables.
4. Split dataset into training and testing sets.
5. Train a Random Forest Classifier.
6. Log parameters and metrics using MLflow.
7. Register the trained model.
8. Upload model artifacts to AWS S3.
9. Download the registered model automatically when the API starts.
10. Serve predictions through FastAPI.

---

## API Endpoints

### Home

```
GET /
```

Response

```json
{
  "message": "Customer Churn Prediction API is Running Successfully"
}
```

---

### Health Check

```
GET /health
```

Response

```json
{
  "status": "healthy"
}
```

---

### Predict Customer Churn

```
POST /predict
```

Request Body

```json
{
  "CustomerID": 1001,
  "Gender": 1,
  "Age": 35,
  "Tenure": 24,
  "MonthlyCharges": 75.5,
  "TotalCharges": 1800.25,
  "ContractType": 1,
  "InternetService": 2,
  "PaymentMethod": 0
}
```

Response

```json
{
  "prediction": 0,
  "result": "Customer Will Stay"
}
```

or

```json
{
  "prediction": 1,
  "result": "Customer Will Churn"
}
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Abhisheknm3896/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file with the following variables:

```
AWS_ACCESS_KEY=YOUR_ACCESS_KEY
AWS_SECRET_KEY=YOUR_SECRET_KEY
AWS_REGION=YOUR_REGION
BUCKET_NAME=YOUR_BUCKET_NAME
```

---

## Train the Model

```bash
python train.py
```

---

## Upload Model to AWS S3

```bash
python upload_to_s3.py
```

---

## Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Docker

Build Docker Image

```bash
docker build -t customer-churn .
```

Run Docker Container

```bash
docker run -p 8000:8000 customer-churn
```

---

## API Documentation

Swagger UI

```
http://localhost:8000/docs
```

ReDoc

```
http://localhost:8000/redoc
```

---

## Future Enhancements

* Hyperparameter tuning
* XGBoost and LightGBM models
* JWT authentication
* CI/CD pipeline using GitHub Actions
* Kubernetes deployment
* Model monitoring and drift detection
* Automated retraining pipeline

---

## Author

**Abhishek N M**

Data Science Student | Python Developer | Machine Learning Enthusiast

GitHub:
https://github.com/Abhisheknm3896

---

## 🌐 Live Demo

The Customer Churn Prediction API is deployed on Render and is publicly accessible.

**Public API URL**

https://customer-churn-prediction-jrs5.onrender.com

### Available Endpoints

| Endpoint   | Method | Description            |
| ---------- | ------ | ---------------------- |
| `/`        | GET    | Home endpoint          |
| `/health`  | GET    | Health check           |
| `/predict` | POST   | Predict customer churn |

### Swagger API Documentation

https://customer-churn-prediction-jrs5.onrender.com/docs

### ReDoc Documentation

https://customer-churn-prediction-jrs5.onrender.com/redoc

## License

This project is developed for educational and learning purposes.
