from fastapi import APIRouter, UploadFile, File, HTTPException
import os, uuid, shutil
from pathlib import Path
import numpy as np
import cv2 as cv

router = APIRouter()

# ===================== PATHS =====================

# backend/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = Path(__file__).resolve().parent

UPLOADS_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model files (downloaded by model_downloader)
MODEL_DIR = PROJECT_ROOT / "models" / "image_colorization"

CAFFEMODEL_PATH = MODEL_DIR / "colorization_release_v2.caffemodel"
PTS_PATH = SERVICE_DIR / "models" / "pts_in_hull.npy"
PROTOTXT_PATH = SERVICE_DIR / "models" / "colorization_deploy_v2.prototxt"

# ===================== VALIDATION =====================

if not CAFFEMODEL_PATH.exists():
    raise RuntimeError("Colorization caffemodel not found. Run model_downloader.")

if not PTS_PATH.exists():
    raise RuntimeError("pts_in_hull.npy missing.")

if not PROTOTXT_PATH.exists():
    raise RuntimeError("prototxt file missing.")

# ===================== LOAD MODEL =====================

pts = np.load(str(PTS_PATH))
net_color = cv.dnn.readNetFromCaffe(
    str(PROTOTXT_PATH),
    str(CAFFEMODEL_PATH)
)

pts = pts.transpose().reshape(2, 313, 1, 1)
net_color.getLayer(net_color.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
net_color.getLayer(net_color.getLayerId("conv8_313_rh")).blobs = [
    np.full([1, 313], 2.606, np.float32)
]

# ===================== COLORIZATION FUNCTION =====================

def colorize_image_file(input_path: str, output_path: str):
    image = cv.imread(input_path)
    if image is None:
        raise ValueError("Invalid image file")

    rgb = image[:, :, ::-1].astype(np.float32) / 255.0
    lab = cv.cvtColor(rgb, cv.COLOR_RGB2Lab)
    l = lab[:, :, 0]

    l_resized = cv.resize(l, (224, 224))
    l_resized -= 50

    net_color.setInput(cv.dnn.blobFromImage(l_resized))
    ab = net_color.forward()[0].transpose((1, 2, 0))
    ab = cv.resize(ab, (image.shape[1], image.shape[0]))

    lab_out = np.concatenate((l[:, :, None], ab), axis=2)
    bgr = cv.cvtColor(lab_out, cv.COLOR_Lab2BGR)
    bgr = np.clip(bgr, 0, 1)

    cv.imwrite(output_path, (bgr * 255).astype("uint8"))

# ===================== API =====================

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    ext = os.path.splitext(file.filename)[1] or ".jpg"
    input_name = f"{uuid.uuid4().hex}{ext}"
    input_path = UPLOADS_DIR / input_name

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_name = f"{uuid.uuid4().hex}.png"
    output_path = RESULTS_DIR / output_name

    try:
        colorize_image_file(str(input_path), str(output_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        input_path.unlink(missing_ok=True)

    return {"result_url": f"/results/{output_name}"}
