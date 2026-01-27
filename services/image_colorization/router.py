from fastapi import APIRouter, UploadFile, File, HTTPException
import os, uuid, shutil
from pathlib import Path
import numpy as np
import cv2 as cv

from utils.model_downloader import download_model

router = APIRouter()

# project root (backend/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = Path(__file__).resolve().parent

# Runtime dirs
UPLOADS_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===================== MODEL SETUP =====================

# Global models directory (gitignored)
MODEL_DIR = PROJECT_ROOT / "models" / "image_colorization"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Google Drive FILE ID (CAFFE MODEL ONLY)
DRIVE_FILE_ID = "1sJ1Rdi1fLNHWC-udvd70FT5WSiL_SON2"

CAFFEMODEL_PATH = MODEL_DIR / "colorization_release_v2.caffemodel"

# Download model if missing
download_model(DRIVE_FILE_ID, str(CAFFEMODEL_PATH))

# Local small files (keep in repo)
PTS_PATH = SERVICE_DIR / "models" / "pts_in_hull.npy"
PROTOTXT_PATH = SERVICE_DIR / "models" / "colorization_deploy_v2.prototxt"

# ===================== LOAD NETWORK =====================

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
