from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import numpy as np
import cv2 as cv

router = APIRouter()

# Project root (backend/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model paths relative to project root
MODEL_DIR = PROJECT_ROOT / "models" / "image_colorization"
CAFFEMODEL_PATH = MODEL_DIR / "colorization_release_v2.caffemodel"
PROTOTXT_PATH = MODEL_DIR / "colorization_deploy_v2.prototxt"
PTS_PATH = MODEL_DIR / "pts_in_hull.npy"

net_color = None  # lazy-loaded model


def load_model():
    global net_color

    if net_color is not None:
        return net_color

    # Check if all files exist
    for file_path in [CAFFEMODEL_PATH, PROTOTXT_PATH, PTS_PATH]:
        if not file_path.exists():
            raise RuntimeError(f"Missing model file: {file_path}")

    pts = np.load(str(PTS_PATH))

    net = cv.dnn.readNetFromCaffe(
        str(PROTOTXT_PATH),
        str(CAFFEMODEL_PATH)
    )

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full([1, 313], 2.606, np.float32)
    ]

    net_color = net
    return net_color


@router.post("/predict")
async def colorize_image(file: UploadFile = File(...)):
    try:
        net = load_model()

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Upload an image file")

        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        gray = cv.imdecode(np_img, cv.IMREAD_GRAYSCALE)

        if gray is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Convert to LAB
        img_rgb = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
        l = img_lab[:, :, 0]

        # Resize and normalize
        l_resized = cv.resize(l, (224, 224))
        l_resized = l_resized - 50

        net.setInput(cv.dnn.blobFromImage(l_resized))
        ab = net.forward()[0].transpose((1, 2, 0))
        ab = cv.resize(ab, (gray.shape[1], gray.shape[0]))

        lab_out = np.concatenate((l[:, :, np.newaxis], ab), axis=2)
        bgr_out = cv.cvtColor(lab_out, cv.COLOR_LAB2BGR)
        bgr_out = np.clip(bgr_out, 0, 1)

        return {
            "message": "Image colorized successfully",
            "height": int(bgr_out.shape[0]),
            "width": int(bgr_out.shape[1])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
