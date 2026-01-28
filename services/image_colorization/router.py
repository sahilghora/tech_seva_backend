from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import numpy as np
import cv2 as cv

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SERVICE_MODEL_DIR = PROJECT_ROOT / "services" / "image_colorization" / "models"
DOWNLOADED_MODEL_DIR = PROJECT_ROOT / "models" / "image_colorization"

CAFFEMODEL_PATH = DOWNLOADED_MODEL_DIR / "colorization_release_v2.caffemodel"
PROTOTXT_PATH = SERVICE_MODEL_DIR / "colorization_deploy_v2.prototxt"
PTS_PATH = SERVICE_MODEL_DIR / "pts_in_hull.npy"

net_color = None


def load_model():
    global net_color

    if net_color is not None:
        return net_color

    for f in [CAFFEMODEL_PATH, PROTOTXT_PATH, PTS_PATH]:
        if not f.exists():
            raise RuntimeError(f"Missing model file: {f}")

    pts = np.load(str(PTS_PATH))

    net = cv.dnn.readNetFromCaffe(
        str(PROTOTXT_PATH),
        str(CAFFEMODEL_PATH)
    )

    # ✅ FIXED BLOBS (THIS WAS YOUR BUG)
    class8_ab = net.getLayerId("class8_ab")
    conv8_313_rh = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8_ab).blobs = [pts.astype("float32")]
    net.getLayer(conv8_313_rh).blobs = [
        np.full((1, 313, 1, 1), 2.606, dtype="float32")
    ]

    net_color = net
    return net_color


@router.post("/predict")
async def colorize_image(file: UploadFile = File(...)):
    try:
        net = load_model()

        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "Upload a valid image")

        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        bgr = cv.imdecode(np_img, cv.IMREAD_COLOR)

        if bgr is None:
            raise HTTPException(400, "Invalid image")

        h, w = bgr.shape[:2]

        img_rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
        l = img_lab[:, :, 0]

        l_resized = cv.resize(l, (224, 224))
        l_resized = l_resized - 50

        blob = cv.dnn.blobFromImage(l_resized)
        net.setInput(blob)

        ab = net.forward()[0].transpose((1, 2, 0))
        ab = cv.resize(ab, (w, h))

        lab_out = np.concatenate((l[:, :, None], ab), axis=2)
        bgr_out = cv.cvtColor(lab_out, cv.COLOR_LAB2BGR)
        bgr_out = np.clip(bgr_out, 0, 1)

        return {
            "message": "Colorization successful",
            "height": h,
            "width": w
        }

    except Exception as e:
        raise HTTPException(500, str(e))
