# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 09:37:00 2025

@author: alexi
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# Initialisation de l'application FastAPI
# -------------------------------------------------
app = FastAPI(
    title="Car Price Prediction API",
    description="Predict vehicle MSRP using a fine-tuned XGBoost model",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # DEV uniquement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Chargement du modèle entraîné (pipeline complet)
# -------------------------------------------------
model = joblib.load("price_model.pkl")

# -------------------------------------------------
# Schéma des données d'entrée (API contract)
# -------------------------------------------------
class CarInput(BaseModel):
    Year: int

    Engine_HP: Optional[float] = None
    Engine_Cylinders: Optional[float] = None
    Number_of_Doors: Optional[float] = None
    highway_MPG: Optional[float] = None
    city_mpg: Optional[float] = None
    Popularity: Optional[int] = None

    Make: str
    Model: str
    Engine_Fuel_Type: str
    Transmission_Type: str
    Driven_Wheels: str
    Vehicle_Size: str
    Vehicle_Style: str


# -------------------------------------------------
# Endpoint de prédiction
# -------------------------------------------------
@app.post("/predict")
def predict_price(car: CarInput):
    """
    Predict the MSRP of a vehicle.
    Missing numerical values are handled by the preprocessing pipeline.
    """

    # Reconstruction EXACTE des colonnes du dataset d'entraînement
    input_df = pd.DataFrame([{
        "Year": car.Year,
        "Engine HP": car.Engine_HP,
        "Engine Cylinders": car.Engine_Cylinders,
        "Number of Doors": car.Number_of_Doors,
        "highway MPG": car.highway_MPG,
        "city mpg": car.city_mpg,
        "Popularity": car.Popularity,

        "Make": car.Make,
        "Model": car.Model,
        "Engine Fuel Type": car.Engine_Fuel_Type,
        "Transmission Type": car.Transmission_Type,
        "Driven_Wheels": car.Driven_Wheels,
        "Vehicle Size": car.Vehicle_Size,
        "Vehicle Style": car.Vehicle_Style
    }])

    # -----------------------------
    # Indicateur de confiance
    # -----------------------------
    optional_fields = [
        car.Engine_HP,
        car.Engine_Cylinders,
        car.Number_of_Doors,
        car.highway_MPG,
        car.city_mpg,
        car.Popularity
    ]

    missing_count = sum(v is None for v in optional_fields)

    if missing_count == 0:
        confidence = "high"
    elif missing_count <= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # -----------------------------
    # Prédiction
    # -----------------------------
    prediction = model.predict(input_df)[0]

    # -----------------------------
    # Réponse API (FORMAT INCHANGÉ)
    # -----------------------------
    return {
        "predicted_price": round(float(prediction), 2),
        "price_min": round(float(prediction * 0.9), 2),
        "price_max": round(float(prediction * 1.1), 2),
        "confidence_level": confidence,
        "missing_features_count": missing_count
    }


# -------------------------------------------------
# Endpoint simple pour vérifier que l'API fonctionne
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "Car Price Prediction API is running"}

