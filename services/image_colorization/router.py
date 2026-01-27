# services/image_colorization/router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import os, uuid, shutil
from pathlib import Path
import numpy as np
import cv2 as cv

router = APIRouter()

# project root (backend/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVICE_DIR / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load colorization model (Caffe)
pts_path = str(MODELS_DIR / "pts_in_hull.npy")
prototxt_path = str(MODELS_DIR / "colorization_deploy_v2.prototxt")
caffemodel_path = str(MODELS_DIR / "colorization_release_v2.caffemodel")

# load network safely
numpy_file = np.load(pts_path)
net_color = cv.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)
net_color.getLayer(net_color.getLayerId('class8_ab')).blobs = [numpy_file.astype(np.float32)]
net_color.getLayer(net_color.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

def colorize_image_file(input_path: str, output_path: str) -> None:
    frame = cv.imread(input_path)
    if frame is None:
        raise ValueError("Could not read input image.")

    rgb_img = (frame[..., ::-1].astype(np.float32) / 255.0)
    lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
    l_channel = lab_img[:, :, 0]
    l_rs = cv.resize(l_channel, (224, 224))
    l_rs -= 50

    net_color.setInput(cv.dnn.blobFromImage(l_rs))
    ab = net_color.forward()[0].transpose((1, 2, 0))
    ab_us = cv.resize(ab, (frame.shape[1], frame.shape[0]))
    lab_out = np.concatenate((l_channel[:, :, np.newaxis], ab_us), axis=2)
    bgr_out = np.clip(cv.cvtColor(lab_out, cv.COLOR_Lab2BGR), 0, 1)

    cv.imwrite(output_path, (bgr_out * 255).astype(np.uint8))

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    ext = os.path.splitext(file.filename)[1] or ".jpg"
    input_filename = f"{uuid.uuid4().hex}{ext}"
    input_path = UPLOADS_DIR / input_filename

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = RESULTS_DIR / output_filename

    try:
        colorize_image_file(str(input_path), str(output_path))
    except Exception as e:
        # cleanup then raise
        if input_path.exists():
            input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    # leave the output in results/ (main.py mounts it at /results)
    return {"result_url": f"/results/{output_filename}"}
