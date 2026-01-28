from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tensorflow as tf
import shutil
import os
from .preprocess import preprocess_image

router = APIRouter()

MODEL_DIR = Path("models/diabetic_retinopathy")
MODEL_PATH = MODEL_DIR / "dr_model.h5"

model = tf.keras.models.load_model(str(MODEL_PATH))

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    temp_path = Path(f"temp_{file.filename}")
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
        "prediction": "Diabetic Retinopathy" if prediction > 0.5 else "No Diabetic Retinopathy",
        "confidence": float(prediction)
    }
