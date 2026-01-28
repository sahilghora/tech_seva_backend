from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tensorflow as tf
import shutil
import os
from .preprocess import preprocess_image

router = APIRouter()

# Project root (backend/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model paths relative to project root
MODEL_DIR = PROJECT_ROOT / "models" / "diabetic_retinopathy"
MODEL_PATH = MODEL_DIR / "dr_model.h5"

model = None  # lazy-loaded model

def load_model():
    global model
    if model is not None:
        return model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model file: {MODEL_PATH}")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    return model

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    temp_path = Path(f"temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        img = preprocess_image(str(temp_path))
        prediction = load_model().predict(img)[0][0]  # lazy-load model
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return {
        "prediction": "Diabetic Retinopathy" if prediction > 0.5 else "No Diabetic Retinopathy",
        "confidence": float(prediction)
    }
