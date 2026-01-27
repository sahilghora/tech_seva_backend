# services/house_price/router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib, os, numpy as np

router = APIRouter()
SERVICE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVICE_DIR / "models"

house_model = joblib.load(MODELS_DIR / "house_price_model.pkl")
scaler_house = joblib.load(MODELS_DIR / "scaler.pkl")

class HouseData(BaseModel):
    number_of_bedrooms: int
    number_of_bathrooms: float
    living_area: float
    lot_area: float
    number_of_floors: float
    waterfront_present: int
    number_of_views: int
    condition_of_the_house: int
    grade_of_the_house: int
    area_excluding_basement: float
    area_of_basement: float
    built_year: int
    renovation_year: int
    postal_code: int
    lattitude: float
    longitude: float
    living_area_renov: float
    lot_area_renov: float
    number_of_schools_nearby: int
    distance_from_airport: float

@router.post("/predict")
def predict_house_price(data: HouseData):
    try:
        data_dict = data.dict()
        arr = np.array(list(data_dict.values())).reshape(1, -1)
        scaled = scaler_house.transform(arr)
        pred = house_model.predict(scaled)[0]
        return {"predicted_price": round(float(pred), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
