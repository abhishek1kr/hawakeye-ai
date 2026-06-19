import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from loguru import logger

# Configuration
WEIGHTS_DIR = Path("models/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "repo_id": "athifsaleem/yolo11m-model",
        "filename": "yolo11m.pt",
        "target": "yolo11m.pt"
    },
    {
        "repo_id": "leeyunjai/yolo11-road-seg",
        "filename": "yolo11m-road-seg.pt",
        "target": "yolo11m-road-seg.pt"
    }

]

def download_all():
    logger.info(f"Starting weight download to {WEIGHTS_DIR}...")
    
    for model in MODELS:
        target_path = WEIGHTS_DIR / model["target"]
        
        # Check if already exists
        if target_path.exists():
            logger.info(f"Checking existing model: {model['target']} ({target_path.stat().st_size / 1024 / 1024:.2f} MB)")
            # Optional: Add logic to force redownload if file is small/corrupt
            if target_path.stat().st_size < 1000: # Clearly too small
                logger.warning(f"File seems too small, re-downloading...")
                target_path.unlink()
            else:
                logger.info(f"Model {model['target']} already exists. Skipping.")
                continue

        logger.info(f"Downloading {model['repo_id']} -> {model['target']}...")
        try:
            download_path = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"]
            )
            # Move to target location
            import shutil
            shutil.copy(download_path, target_path)
            logger.info(f"Successfully saved to {target_path}")
        except Exception as e:
            logger.error(f"Failed to download {model['repo_id']}: {e}")

if __name__ == "__main__":
    download_all()
    logger.info("All model weights are ready!")
