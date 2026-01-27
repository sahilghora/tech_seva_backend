# services/fraud_transaction/router.py
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import joblib, json, pandas as pd, traceback
import random
import logging

router = APIRouter()

logger = logging.getLogger(__name__)


SERVICE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVICE_DIR / "models"

MODEL_PATH = MODELS_DIR / "catboost_fraud_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler1.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"
ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
DEVICE_COLS_PATH = MODELS_DIR / "device_used_columns.json"
THRESHOLD_PATH = MODELS_DIR / "high_amount_threshold.json"

numeric_cols = [
    'Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days',
    'Transaction Hour', 'Transaction Weekday', 'Transaction Month',
    'Amount_per_AccountDay', 'Total_Purchase_Value'
]


def load_artifact(path: Path, name: str):
    if not path.is_file():
        raise RuntimeError(f"Missing {name} at: {path}")
    return joblib.load(path)

def load_json(path: Path, name: str):
    if not path.is_file():
        raise RuntimeError(f"Missing {name} at: {path}")
    with open(path) as f:
        return json.load(f)


try:
    model = load_artifact(MODEL_PATH, "Fraud model")
    scaler = load_artifact(SCALER_PATH, "Scaler")
    label_encoders = load_artifact(ENCODERS_PATH, "Label encoders")
    feature_columns = load_json(FEATURES_PATH, "Feature columns")
    device_used_columns = load_json(DEVICE_COLS_PATH, "Device columns")
    high_amount_threshold = load_json(THRESHOLD_PATH, "High amount threshold")["threshold"]
    print("✅ Fraud Transaction artifacts loaded successfully")
except Exception as e:
    print(f" Error loading fraud_transaction artifacts: {e}")


class TransactionInput(BaseModel):
    transaction_date: str
    transaction_amount: float
    quantity: int
    customer_age: int
    account_age_days: int
    shipping_address: str
    billing_address: str
    payment_method: str
    product_category: str
    customer_location: str
    device_used: str


def safe_encode(encoder, value, field):
    # Normalize input
    norm_value = value.lower().replace(" ", "_")

    # 1️⃣ Exact match
    if norm_value in encoder.classes_:
        return encoder.transform([norm_value])[0]

    # 2️⃣ Closest / partial match
    for cls in encoder.classes_:
        if norm_value in cls or cls in norm_value:
            logger.warning(
                f"{field}: '{value}' not found, mapped to closest '{cls}'"
            )
            return encoder.transform([cls])[0]

    # 3️⃣ Random fallback
    fallback = random.choice(list(encoder.classes_))
    logger.warning(
        f"{field}: '{value}' not found, using random fallback '{fallback}'"
    )

    return encoder.transform([fallback])[0]

def build_input_df(inp: TransactionInput) -> pd.DataFrame:
    dt = pd.to_datetime(inp.transaction_date)

    row = {
        "Transaction Amount": inp.transaction_amount,
        "Quantity": inp.quantity,
        "Customer Age": inp.customer_age,
        "Account Age Days": inp.account_age_days,
        "Transaction Hour": dt.hour,
        "Transaction Weekday": dt.weekday(),
        "Transaction Month": dt.month,
        "Address Mismatch": int(inp.shipping_address != inp.billing_address),
        "High Amount": int(inp.transaction_amount > high_amount_threshold)
    }

    row["Amount_per_AccountDay"] = row["Transaction Amount"] / (row["Account Age Days"] + 1)
    row["Total_Purchase_Value"] = row["Transaction Amount"] * row["Quantity"]

    # Safe label encoding (with fallback)
    row["Payment Method"] = safe_encode(
        label_encoders["Payment Method"],
        inp.payment_method,
        "payment_method"
    )
    row["Product Category"] = safe_encode(
        label_encoders["Product Category"],
        inp.product_category,
        "product_category"
    )
    row["Customer Location"] = safe_encode(
        label_encoders["Customer Location"],
        inp.customer_location,
        "customer_location"
    )

    df = pd.DataFrame([row])

    # One-hot encode devices
    device_df = pd.DataFrame(columns=device_used_columns)
    for col in device_used_columns:
        device_df.loc[0, col] = 1 if col == f"Device Used_{inp.device_used}" else 0

    df = pd.concat([df, device_df], axis=1)

    # Ensure all features exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder & scale numeric
    df = df[feature_columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


@router.post("/predict")
def predict(input: TransactionInput, threshold: Optional[float] = Query(0.3, ge=0.0, le=1.0)):
    try:
        X = build_input_df(input)
        prob = model.predict_proba(X)[:, 1][0]
        return {
            "is_fraud": int(prob > threshold),
            "probability": float(prob),
            "threshold": float(threshold)
        }
    except Exception as e:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error: {e}\n{traceback_str}")
