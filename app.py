import os
import numpy as np
import mlflow.pyfunc

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field

from download_from_s3 import download_registered_model

# Download model from S3 if not available locally
download_registered_model()

# Load MLflow model
model = mlflow.pyfunc.load_model("registered_model/models/artifacts")

print("Model Loaded Successfully")

# Create FastAPI App
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn using a trained MLflow model.",
    version="1.0"
)


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


# Health Endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction Endpoint
@app.post("/predict")
def predict(customer: Customer):

    try:

        data = [[
            customer.CustomerID,
            customer.Gender,
            customer.Age,
            customer.Tenure,
            customer.MonthlyCharges,
            customer.TotalCharges,
            customer.ContractType,
            customer.InternetService,
            customer.PaymentMethod
        ]]

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