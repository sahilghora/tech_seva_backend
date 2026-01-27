# services/diabetic_retinopathy/router.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tensorflow as tf
import shutil
import os

from .preprocess import preprocess_image

router = APIRouter()

# ===================== PATH SETUP =====================

# backend/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "diabetic_retinopathy"
MODEL_PATH = MODEL_DIR / "dr_model.h5"

SERVICE_DIR = Path(__file__).resolve().parent

# ===================== LOAD MODEL =====================

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Diabetic Retinopathy model not found. "
        "Ensure models are downloaded during app startup."
    )

model = tf.keras.models.load_model(str(MODEL_PATH))

# ===================== ROUTE =====================

@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    temp_path = SERVICE_DIR / f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        img = preprocess_image(str(temp_path))
        prediction = model.predict(img)[0][0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return {
        "prediction": (
            "Diabetic Retinopathy"
            if prediction > 0.5
            else "No Diabetic Retinopathy"
        ),
        "confidence": round(float(prediction), 4)
    }
