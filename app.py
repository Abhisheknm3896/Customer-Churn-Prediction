# Customer Churn Prediction API using FastAPI
import os
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# Create FastAPI App

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn using a trained Random Forest model.",
    version="1.0"
)

# Load Trained Model
MODEL_PATH = "models/churn_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. Please run train.py first."
    )

model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully")

# Input Schema

class Customer(BaseModel):
    CustomerID: int
    Gender: int
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0)
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., gt=0)
    ContractType: int
    InternetService: int
    PaymentMethod: int

# Home Endpoint

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running Successfully"
    }

# Health Check Endpoint

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction Endpoint
@app.post("/predict")
def predict(customer: Customer):

    try:
        data = np.array([[
            customer.CustomerID,
            customer.Gender,
            customer.Age,
            customer.Tenure,
            customer.MonthlyCharges,
            customer.TotalCharges,
            customer.ContractType,
            customer.InternetService,
            customer.PaymentMethod
        ]])

        prediction = model.predict(data)

        result = (
            "Customer Will Churn"
            if prediction[0] == 1
            else "Customer Will Stay"
        )

        return {
            "prediction": int(prediction[0]),
            "result": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )