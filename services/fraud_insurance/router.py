# services/fraud_insurance/router.py
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import pandas as pd
import pickle, traceback
import random

router = APIRouter()
SERVICE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVICE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fraud_model_top_features.pkl"
ENCODERS_PATH = MODELS_DIR / "label_encoders_insurance.pkl"
TOP_FEATURES_PATH = MODELS_DIR / "top_features.pkl"

categorical_cols = [
    "policy_state",
    "policy_csl",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "auto_make",
    "auto_model"
]


if not MODEL_PATH.exists():
    raise RuntimeError("Missing model file")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    label_encoders = pickle.load(f)

with open(TOP_FEATURES_PATH, "rb") as f:
    top_features = pickle.load(f)


class InsuranceInput(BaseModel):
    months_as_customer: int
    age: int
    policy_state: str
    policy_csl: str
    policy_deductable: int
    policy_annual_premium: float
    umbrella_limit: int
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    vehicle_claim: float
    auto_make: str
    auto_model: str
    auto_year: int
    incident_month: int
    incident_day_of_week: int
    injury_ratio: float
    property_ratio: float
    vehicle_ratio: float


def safe_encode(encoder, value, column_name: str):
    """
    Safely encode categorical values.
    Handles unseen labels without crashing.
    """
    try:
        value = str(value).strip()

        # exact match
        if value in encoder.classes_:
            return encoder.transform([value])[0]

        # case-insensitive match
        lower_map = {c.lower(): c for c in encoder.classes_}
        if value.lower() in lower_map:
            return encoder.transform([lower_map[value.lower()]])[0]

        # fallback → random known class
        fallback = random.choice(list(encoder.classes_))
        return encoder.transform([fallback])[0]

    except Exception:
        # ultimate fallback
        return 0

def build_input_df(inp: InsuranceInput) -> pd.DataFrame:
    row = inp.dict()

    for col in categorical_cols:
        if col in row and col in label_encoders:
            row[col] = safe_encode(label_encoders[col], row[col], col)

    df = pd.DataFrame([row])

    # ensure all top features exist
    for col in top_features:
        if col not in df.columns:
            df[col] = 0

    df = df[top_features]
    return df

@router.post("/predict")
def predict(
    input: InsuranceInput,
    threshold: Optional[float] = Query(0.3, ge=0.0, le=1.0)
):
    try:
        X = build_input_df(input)
        prob = model.predict_proba(X)[:, 1][0]

        return {
            "is_fraud": int(prob > threshold),
            "fraud_probability": float(prob),
            "threshold": float(threshold)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {e}\n{traceback.format_exc()}"
        )
