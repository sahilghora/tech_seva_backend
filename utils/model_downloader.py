import os
import gdown

def download_model(file_id: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(output_path):
        print(f"📥 Downloading model to {output_path}")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"✅ Model already exists at {output_path}")
