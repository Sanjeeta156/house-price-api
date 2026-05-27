from fastapi import FastAPI
from app.schema import HouseData
import joblib
import numpy as np

app = FastAPI(title="House Price Prediction API")

model = joblib.load("app/model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict(data: HouseData):

    features = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    prediction = model.predict(features)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }