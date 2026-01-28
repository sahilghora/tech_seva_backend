from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import numpy as np
import cv2 as cv

router = APIRouter()

MODEL_DIR = Path("models/image_colorization")
CAFFEMODEL_PATH = MODEL_DIR / "colorization_release_v2.caffemodel"
SERVICE_DIR = Path(__file__).resolve().parent

# Local small files
PTS_PATH = SERVICE_DIR / "models/pts_in_hull.npy"
PROTOTXT_PATH = SERVICE_DIR / "models/colorization_deploy_v2.prototxt"

# Load network
pts = np.load(str(PTS_PATH))
net_color = cv.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
pts = pts.transpose().reshape(2, 313, 1, 1)
net_color.getLayer(net_color.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
net_color.getLayer(net_color.getLayerId("conv8_313_rh")).blobs = [
    np.full([1, 313], 2.606, np.float32)
]
