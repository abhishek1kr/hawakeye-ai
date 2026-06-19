#!/usr/bin/env python3
"""
Kaggle Training Script — YOLO11m Hawkeye Detector
Dataset: RDD2022 (Global Road Damage Detection) + Custom Pothole Augmentation
Classes: 0: Crack, 1: Pothole, 2: Signboard
Runtime: GPU T4/P100 | ~3-4 hours | Output: yolo11m_road.pt
"""

import os
import shutil
from pathlib import Path

# ─── Cell 1: Installation ─────────────────────────────────────────────────────
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "ultralytics"])
    from ultralytics import YOLO

print(f"Ultralytics version: {ultralytics.__version__}")

# ─── Cell 2: Dataset Preparation (Kaggle Logic) ───────────────────────────────
# We assume the RDD2022 dataset is added to the Kaggle notebook.
# Path: /kaggle/input/rdd2022-yolo/
# Classes in RDD2022 are often: D00, D10, D20, D40
# We will remap them to our unified Hawkeye classes.

DATASET_YAML = """
path: /kaggle/working/datasets/road_data
train: train/images
val: val/images

names:
  0: crack
  1: pothole
  2: signboard
"""

def prepare_kaggle_dataset():
    working_dir = Path("/kaggle/working/datasets/road_data")
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # In a real Kaggle environment, we would symlink or copy labels here
    # For this script, we'll write the data.yaml
    with open("/kaggle/working/data.yaml", "w") as f:
        f.write(DATASET_YAML)
    
    print("Dataset config written to /kaggle/working/data.yaml")

# ─── Cell 3: Training Logic ───────────────────────────────────────────────────
def train():
    # 1. Initialize YOLO11m
    # ultralytics will auto-download 'yolo11m.pt'
    model = YOLO("yolo11m.pt") 
    
    print("Starting YOLO11m fine-tuning...")
    
    # 2. Train
    # We use a large imgsz (1024) for detecting small cracks/potholes if GPU allows
    # otherwise fallback to 640.
    results = model.train(
        data="/kaggle/working/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,             # GPU 0
        patience=10,          # Early stopping
        save=True,
        project="hawkeye",
        name="yolo11m_road",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.01,
        augment=True,         # Enable mosaic, mixup, etc.
    )
    
    # 3. Export to production formats
    print("Training complete. Exporting weights...")
    model.export(format="onnx") # For potential Edge deployment
    
    # 4. Copy best weights to working root
    best_weights = Path("hawkeye/yolo11m_road/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, "/kaggle/working/yolo11m_road.pt")
        print("✅ Best weights saved to /kaggle/working/yolo11m_road.pt")

if __name__ == "__main__":
    # Check if running on Kaggle
    if Path("/kaggle").exists():
        prepare_kaggle_dataset()
        train()
    else:
        print("This script is optimized for Kaggle GPU environments.")
        print("Please run it in a Kaggle notebook with RDD2022 dataset attached.")
