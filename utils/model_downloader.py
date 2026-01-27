# utils/model_downloader.py
import os
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_LINKS = {
    "diabetic_retinopathy": "https://drive.google.com/uc?id=1EnbFeLFYPjKH7zSWr5z9PpB5A_rcjwpa",
    "image_colorization": "https://drive.google.com/uc?id=1sJ1Rdi1fLNHWC-udvd70FT5WSiL_SON2",
    "house_price": "https://drive.google.com/uc?id=16Y0dJwFAhfQBpGR1lV8HL5LRip-RzsgV",
}

MODEL_DIRS = {
    "diabetic_retinopathy": PROJECT_ROOT / "services/diabetic_retinopathy",
    "image_colorization": PROJECT_ROOT / "services/image_colorization/models",
    "house_price": PROJECT_ROOT / "services/house_price/models",
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
    # House price model
    download_file(MODEL_LINKS["house_price"], MODEL_DIRS["house_price"] / "house_price_model.pkl")

    # Diabetic retinopathy model
    download_file(MODEL_LINKS["diabetic_retinopathy"], MODEL_DIRS["diabetic_retinopathy"] / "dr_model.h5")

    # Image colorization models
    download_file(MODEL_LINKS["image_colorization"], MODEL_DIRS["image_colorization"] / "colorization_release_v2.caffemodel")

    print("All models downloaded successfully!")
