# services/stock_prediction/router.py
import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import xgboost as xgb
from services.stock_prediction.models.feature_engineering import feature_engineering

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class StockData(BaseModel):
    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Adj_Close: float
    Volume: float

def prepare_last_row(data: List[StockData]):
    df = pd.DataFrame([d.dict() for d in data])
    df_processed = feature_engineering(df)
    if df_processed.empty:
        raise HTTPException(status_code=400, detail="Feature engineering returned empty DataFrame.")
    if 'Adj_Close' in df_processed.columns:
        df_processed = df_processed.rename(columns={'Adj_Close': 'Adj Close'})
    X = df_processed.drop(columns=["Date"], errors='ignore')
    last_row = X.iloc[-1].values.reshape(1, -1)
    return df_processed, last_row

model_default = xgb.XGBRegressor()
model_default.load_model(os.path.join(MODELS_DIR, "xgb_best_model.json"))

model_tata_motors = xgb.XGBRegressor()
model_tata_motors.load_model(os.path.join(MODELS_DIR, "xgb_best_model1.json"))

model_tata_steel = xgb.XGBRegressor()
model_tata_steel.load_model(os.path.join(MODELS_DIR, "xgb_best_model2.json"))

model_tata_power = xgb.XGBRegressor()
model_tata_power.load_model(os.path.join(MODELS_DIR, "xgb_best_model3.json"))


@router.post("/predict/default")
def predict_default(data: List[StockData]):
    if len(data) < 20:
        raise HTTPException(status_code=400, detail="Provide at least 20 rows of stock data.")
    df_processed, last_row = prepare_last_row(data)
    pred = model_default.predict(last_row)[0]
    return {"Date": df_processed["Date"].iloc[-1], "predicted_close": float(pred)}

@router.post("/predict/tata_motors")
def predict_tata_motors(data: List[StockData]):
    if len(data) < 20:
        raise HTTPException(status_code=400, detail="Provide at least 20 rows of stock data.")
    df_processed, last_row = prepare_last_row(data)
    pred = model_tata_motors.predict(last_row)[0]
    return {"Date": df_processed["Date"].iloc[-1], "predicted_close": float(pred)}

@router.post("/predict/tata_steel")
def predict_tata_steel(data: List[StockData]):
    if len(data) < 20:
        raise HTTPException(status_code=400, detail="Provide at least 20 rows of stock data.")
    df_processed, last_row = prepare_last_row(data)
    pred = model_tata_steel.predict(last_row)[0]
    return {"Date": df_processed["Date"].iloc[-1], "predicted_close": float(pred)}

@router.post("/predict/tata_power")
def predict_tata_power(data: List[StockData]):
    if len(data) < 20:
        raise HTTPException(status_code=400, detail="Provide at least 20 rows of stock data.")
    df_processed, last_row = prepare_last_row(data)
    pred = model_tata_power.predict(last_row)[0]
    return {"Date": df_processed["Date"].iloc[-1], "predicted_close": float(pred)}
