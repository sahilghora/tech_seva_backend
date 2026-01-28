from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np

router = APIRouter()

# backend/
BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "models" / "house_price"
HOUSE_MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# 🔁 Lazy-loaded globals
house_model = None
scaler_house = None


def load_models():
    global house_model, scaler_house

    if house_model is None or scaler_house is None:
        if not HOUSE_MODEL_PATH.exists():
            raise RuntimeError("House price model not found. Downloader failed.")

        if not SCALER_PATH.exists():
            raise RuntimeError("House price scaler not found. Downloader failed.")

        house_model = joblib.load(HOUSE_MODEL_PATH)
        scaler_house = joblib.load(SCALER_PATH)


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
        # ✅ Ensure models are loaded AFTER startup downloader
        load_models()

        arr = np.array(list(data.dict().values())).reshape(1, -1)
        scaled = scaler_house.transform(arr)
        pred = house_model.predict(scaled)[0]

        return {"predicted_price": round(float(pred), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
