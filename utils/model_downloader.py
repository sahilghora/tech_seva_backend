from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_LINKS = {
    "house_price": "https://drive.google.com/uc?export=download&id=16Y0dJwFAhfQBpGR1lV8HL5LRip-RzsgV",
    "diabetic_retinopathy": "https://drive.google.com/uc?export=download&id=1EnbFeLFYPjKH7zSWr5z9PpB5A_rcjwpa",
    "image_colorization": "https://drive.google.com/uc?export=download&id=1sJ1Rdi1fLNHWC-udvd70FT5WSiL_SON2",
}

MODEL_PATHS = {
    "house_price": PROJECT_ROOT / "services/house_price/models/house_price_model.pkl",
    "diabetic_retinopathy": PROJECT_ROOT / "services/diabetic_retinopathy/dr_model.h5",
    "image_colorization": PROJECT_ROOT / "services/image_colorization/models/colorization_release_v2.caffemodel",
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
    for key in MODEL_LINKS:
        download_file(MODEL_LINKS[key], MODEL_PATHS[key])

    print("All models downloaded successfully!")
