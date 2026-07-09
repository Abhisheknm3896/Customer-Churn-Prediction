# This script trains a machine learning model to predict customer churn.
# It loads data, prepares it, trains a Random Forest model, and saves the result.

import os  # Used to create folders such as the models directory

import joblib  # Used to save the trained model to a file
import mlflow  # Used to track experiments and model training
import mlflow.sklearn  # MLflow support for scikit-learn models
import pandas as pd  # Used to read and work with the dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    # Load the dataset from the CSV file
    df = pd.read_csv("data/churn.csv")

    print("Dataset Loaded Successfully")
    print(df.head())

    # Create the models folder if it does not already exist
    os.makedirs("models", exist_ok=True)

    # Remove rows with missing values
    df = df.dropna()

    # Convert categorical/text columns into numeric values
    encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    print("\nData After Encoding")
    print(df.head())

    # Separate features (X) from the target label (y)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Start an MLflow experiment to track the run
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run():
        # Create and train the Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate model performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\nAccuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 Score :", f1)

        # Log model details and metrics to MLflow
        mlflow.log_param("Model", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        #  Save the trained model and log it to MLflow
        mlflow.sklearn.log_model(model, "Customer_Churn_Model")
        joblib.dump(model, "models/churn_model.joblib")

    print("\nModel Saved Successfully")


if __name__ == "__main__":
    main()
