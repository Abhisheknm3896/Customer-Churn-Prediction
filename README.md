# Customer Churn Prediction

## Overview
This project predicts customer churn using a machine learning model.

## Features
- Train model using MLflow
- Register MLflow model
- Upload model artifacts to Amazon S3
- FastAPI prediction API
- Dockerized application
- AWS Elastic Beanstalk deployment
- GitHub Actions CI/CD

## Tech Stack
- Python
- FastAPI
- MLflow
- Scikit-learn
- Docker
- AWS S3
- AWS Elastic Beanstalk
- GitHub Actions

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Docker

```bash
docker build -t customer-churn .
docker run --env-file .env -p 8000:8000 customer-churn
```

## API

- `GET /`
- `GET /health`
- `POST /predict`