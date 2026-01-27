from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tensorflow as tf
import shutil, os

from .preprocess import preprocess_image
from utils.model_downloader import download_model

router = APIRouter()

SERVICE_DIR = Path(__file__).resolve().parent

# Store model OUTSIDE services (gitignored)
MODEL_DIR = Path("models/diabetic_retinopathy")
MODEL_PATH = MODEL_DIR / "dr_model.h5"

# Google Drive FILE ID
DRIVE_FILE_ID = "1EnbFeLFYPjKH7zSWr5z9PpB5A_rcjwpa"

# Download model if not present
download_model(DRIVE_FILE_ID, str(MODEL_PATH))

# Load model once
model = tf.keras.models.load_model(str(MODEL_PATH))


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
        "prediction": "Diabetic Retinopathy" if prediction > 0.5 else "No Diabetic Retinopathy",
        "confidence": float(prediction)
    }
