# This is a FastAPI web application that provides a prediction endpoint for customer churn.
# It loads the trained model and uses it to predict whether a customer will churn.

from fastapi import FastAPI, HTTPException  # FastAPI framework for building web APIs
from pydantic import BaseModel, Field  # Used to define and validate input data structure
import joblib  # Used to load the trained machine learning model
import numpy as np  # Used for numeric operations and arrays

# Create a FastAPI instance with metadata
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn",
    version="1.0"
)

# Load the trained model from the saved file
model = joblib.load("models/churn_model.joblib")


# Define the input data structure using Pydantic BaseModel
# This ensures the API receives valid customer data
class Customer(BaseModel):
    CustomerID: int
    Gender: int
    Age: int = Field(..., ge=18, le=100)  # Age must be between 18 and 100
    Tenure: int = Field(..., ge=0)  # Tenure must be 0 or greater
    MonthlyCharges: float = Field(..., gt=0)  # Monthly charges must be greater than 0
    TotalCharges: float = Field(..., gt=0)  # Total charges must be greater than 0
    ContractType: int
    InternetService: int
    PaymentMethod: int


# Define a home endpoint that runs when the API starts
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running"
    }


# Define the prediction endpoint that accepts customer data
@app.post("/predict")
def predict(customer: Customer):

    try:
        # Convert customer data into a numpy array for the model
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

        #  Use the trained model to make a prediction
        prediction = model.predict(data)

        # Convert the numeric prediction to a readable message
        if prediction[0] == 1:
            result = "Customer Will Churn"  # 1 means the customer will churn
        else:
            result = "Customer Will Stay"   # 0 means the customer will not churn

        # Return the prediction result to the client
        return {
            "prediction": result
        }

    # Handle any errors that occur during prediction
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )