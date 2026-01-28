# utils/model_downloader.py

import os
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # backend/

# Google Drive direct download links
MODEL_LINKS = {
    "diabetic_retinopathy": "https://drive.google.com/uc?id=1EnbFeLFYPjKH7zSWr5z9PpB5A_rcjwpa",
    "image_colorization": "https://drive.google.com/uc?id=1sJ1Rdi1fLNHWC-udvd70FT5WSiL_SON2",
    "house_price_model": "https://drive.google.com/uc?id=16Y0dJwFAhfQBpGR1lV8HL5LRip-RzsgV",
    "house_price_scaler": "https://drive.google.com/uc?id=YOUR_SCALER_FILE_ID"  # replace with actual
}

# Where to save the models
MODEL_DIRS = {
    "diabetic_retinopathy": PROJECT_ROOT / "models/diabetic_retinopathy",
    "image_colorization": PROJECT_ROOT / "models/image_colorization",
    "house_price_model": PROJECT_ROOT / "models/house_price",
    "house_price_scaler": PROJECT_ROOT / "models/house_price",
}

MODEL_FILES = {
    "diabetic_retinopathy": "dr_model.h5",
    "image_colorization": "colorization_release_v2.caffemodel",
    "house_price_model": "house_price_model.pkl",
    "house_price_scaler": "scaler.pkl",
}

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{dest} already exists, skipping download.")
        return

    print(f"Downloading {dest.name} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {dest.name}")

def download_all_models():
    for key, url in MODEL_LINKS.items():
        dest_file = MODEL_DIRS[key] / MODEL_FILES[key]
        download_file(url, dest_file)
    print("All models downloaded successfully!")
