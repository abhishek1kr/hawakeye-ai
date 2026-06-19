"""
Download Best Pothole Detection + Segmentation Model
Model: keremberke/yolov8n-pothole-segmentation
mAP@0.5 (box)  = 0.995
mAP@0.5 (mask) = 0.995

This model gives BOTH bounding boxes AND pixel-level segmentation masks,
which lets Hawkeye calculate the real pothole area & shape far more accurately.
"""
import sys
import os
from pathlib import Path

# Make sure we run from backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
os.chdir(BACKEND_DIR)

WEIGHTS_DIR = BACKEND_DIR / "models" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PATH = WEIGHTS_DIR / "pothole_seg_yolov8n.pt"

def download():
    print("=" * 60)
    print("  Hawkeye AI -- Pothole Model Downloader")
    print("  Model : keremberke/yolov8n-pothole-segmentation")
    print("  mAP@0.5 (box+mask) : 0.995")
    print("=" * 60)

    if TARGET_PATH.exists():
        size_mb = TARGET_PATH.stat().st_size / (1024 * 1024)
        print("[OK] Model already exists:", TARGET_PATH, f"({size_mb:.1f} MB)")
        print("    Delete the file and re-run to force re-download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("\n[!] huggingface_hub not installed. Running: pip install huggingface_hub")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    print("\n[>>] Downloading from Hugging Face...")
    try:
        # Primary: try to get the best.pt weights directly
        local_path = hf_hub_download(
            repo_id="keremberke/yolov8n-pothole-segmentation",
            filename="best.pt",
            local_dir=str(WEIGHTS_DIR),
            local_dir_use_symlinks=False,
        )
        # Rename to our standard name
        src = Path(local_path)
        if src != TARGET_PATH:
            src.rename(TARGET_PATH)
        print(f"\n[OK] Downloaded --> {TARGET_PATH}")
    except Exception as e:
        print(f"\n[!] Direct download failed: {e}")
        print("    Trying ultralytics hub fallback...")
        try:
            from ultralytics import YOLO
            model = YOLO("hf://keremberke/yolov8n-pothole-segmentation/best.pt")
            # Save the model to our weights directory
            model.save(str(TARGET_PATH))
            print(f"\n[OK] Downloaded via ultralytics --> {TARGET_PATH}")
        except Exception as e2:
            print(f"\n[X] All download methods failed: {e2}")
            print("    Manual download:")
            print("    https://huggingface.co/keremberke/yolov8n-pothole-segmentation/resolve/main/best.pt")
            print("    Save it to:", TARGET_PATH)
            sys.exit(1)

    # Quick validation
    print("\n[*] Validating downloaded model...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(TARGET_PATH))
        print(f"    Model type  : {model.task}")
        print(f"    Model names : {model.names}")
        print(f"[OK] Validation passed!")
    except Exception as e:
        print(f"[!] Validation failed (model may still work): {e}")

    size_mb = TARGET_PATH.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Ready! Model saved: {TARGET_PATH}  ({size_mb:.1f} MB)")
    print("\n    model_config.yaml is already updated to use this model.")
    print("    Restart the backend server and enjoy improved pothole detection!\n")


if __name__ == "__main__":
    download()
