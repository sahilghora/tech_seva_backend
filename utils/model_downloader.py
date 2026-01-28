from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # backend/

# Only models that are NOT in repo and need Google Drive download
MODEL_LINKS = {
    "diabetic_retinopathy": "1EnbFeLFYPjKH7zSWr5z9PpB5A_rcjwpa",
    "image_colorization": "1sJ1Rdi1fLNHWC-udvd70FT5WSiL_SON2",
    "house_price_model": "16Y0dJwFAhfQBpGR1lV8HL5LRip-RzsgV",
}

# Local paths for all models
MODEL_DIRS = {
    "diabetic_retinopathy": PROJECT_ROOT / "models/diabetic_retinopathy",
    "image_colorization": PROJECT_ROOT / "models/image_colorization",
    "house_price_model": PROJECT_ROOT / "models/house_price",
}

MODEL_FILES = {
    "diabetic_retinopathy": "dr_model.h5",
    "image_colorization": "colorization_release_v2.caffemodel",
    "house_price_model": "house_price_model.pkl",
}

# List of models that exist locally and should not be downloaded
LOCAL_MODELS = [
    "house_price_model"  # Example: you already pushed the scaler locally
]


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def download_file(file_id: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"{dest.name} already exists, skipping download.")
        return

    print(f"Downloading {dest.name} ...")

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        response = session.get(
            URL, params={"id": file_id, "confirm": token}, stream=True
        )

    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    print(f"Downloaded {dest.name}")


def download_all_models():
    for key, file_id in MODEL_LINKS.items():
        # Skip local models
        if key in LOCAL_MODELS:
            print(f"{MODEL_FILES[key]} is local, skipping download.")
            continue

        dest_file = MODEL_DIRS[key] / MODEL_FILES[key]
        download_file(file_id, dest_file)

    print("✅ All required models downloaded successfully!")
