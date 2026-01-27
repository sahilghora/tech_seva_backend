# utils/model_downloader.py
import requests
from pathlib import Path

def download_file(url, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def download_all_models():
    # Diabetic Retinopathy
    dr_url = "https://drive.google.com/uc?id=1EnbFeLFYPjKH7zSWr5z9PpB5A_rcjwpa"
    dr_path = Path(__file__).resolve().parent.parent / "services/diabetic_retinopathy/dr_model.h5"
    download_file(dr_url, dr_path)

    # Image Colorization
    color_url = "https://drive.google.com/uc?id=1sJ1Rdi1fLNHWC-udvd70FT5WSiL_SON2"
    color_path = Path(__file__).resolve().parent.parent / "services/image_colorization/models/colorization_release_v2.caffemodel"
    download_file(color_url, color_path)
    
    print("All models downloaded successfully!")
