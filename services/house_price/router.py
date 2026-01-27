# services/house_price/router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np

from utils.model_downloader import download_all_models

router = APIRouter()

SERVICE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVICE_DIR / "models"

house_model = None
scaler_house = None


@router.on_event("startup")
def load_house_price_model():
    global house_model, scaler_house

    # Download models
    download_all_models()

    model_path = MODELS_DIR / "house_price_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if not model_path.exists():
        raise RuntimeError(f"Missing model: {model_path}")

    if not scaler_path.exists():
        raise RuntimeError(f"Missing scaler: {scaler_path}")

    house_model = joblib.load(model_path)
    scaler_house = joblib.load(scaler_path)

    print("✅ House price model loaded")


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
    if house_model is None or scaler_house is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    arr = np.array(list(data.dict().values())).reshape(1, -1)
    scaled = scaler_house.transform(arr)
    pred = house_model.predict(scaled)[0]

    return {"predicted_price": round(float(pred), 2)}
