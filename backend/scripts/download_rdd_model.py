import os
import requests
from pathlib import Path

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

if __name__ == "__main__":
    weights_dir = Path("models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # RDD2022 YOLOv8 model
    model_url = "https://huggingface.co/ozair23/yolov8-road-damage-detector/resolve/main/best.pt"
    dest = weights_dir / "road_damage_yolov8.pt"
    if not dest.exists():
        download_file(model_url, dest)
    else:
        print(f"Model already exists at {dest}")
