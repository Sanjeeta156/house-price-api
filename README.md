# House Price Prediction API

A machine learning API that predicts house prices using FastAPI, XGBoost, and Docker.

## Live Demo
https://house-price-api-tpbc.onrender.com/docs

## Features
- Real-time house price prediction
- FastAPI backend
- Swagger API docs
- Dockerized deployment
- Public cloud hosting on Render

## Tech Stack
- Python
- FastAPI
- Scikit-learn
- XGBoost
- Docker
- Render

## API Endpoint

POST /predict

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Docker Run

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```
## Architecture Diagram

```text
Client/User
     ↓
FastAPI Backend
     ↓
XGBoost ML Model
     ↓
Prediction Response
```
