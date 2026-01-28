from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import numpy as np
import cv2 as cv

router = APIRouter()

# Base directory of this service
SERVICE_DIR = Path(__file__).resolve().parent

# Correct absolute paths
MODEL_DIR = SERVICE_DIR / "models"
CAFFEMODEL_PATH = MODEL_DIR / "colorization_release_v2.caffemodel"
PROTOTXT_PATH = MODEL_DIR / "colorization_deploy_v2.prototxt"
PTS_PATH = MODEL_DIR / "pts_in_hull.npy"

net_color = None  # lazy-loaded model


def load_model():
    global net_color

    if net_color is not None:
        return net_color

    if not CAFFEMODEL_PATH.exists():
        raise RuntimeError(f"Missing model file: {CAFFEMODEL_PATH}")

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
    return net
