from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn",
    version="1.0"
)

# Load trained model
model = joblib.load("models/churn_model.joblib")


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


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running"
    }


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

        if prediction[0] == 1:
            result = "Customer Will Churn"
        else:
            result = "Customer Will Stay"

        return {
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )