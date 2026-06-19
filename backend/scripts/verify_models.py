import torch
import cv2
import numpy as np
import traceback
from loguru import logger
from pathlib import Path
from src.detection import YOLO11Detector, CrackDetector, RoadSurfaceClassifier
from src.segmentation import RoadYOLO

def verify_all_models():
    logger.info("🚀 Starting Hawkeye AI Model Verification...")
    
    # Create a dummy frame (640x640)
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # 1. Verify YOLO11 Pothole Detector (athifsaleem)
    logger.info("--- 1. Testing YOLO11 Pothole Detector (athifsaleem/yolo11m-model) ---")
    try:
        pothole_det = YOLO11Detector(task="detect") # Default weights in code is athifsaleem
        dets = pothole_det.detect(dummy_frame)
        logger.success(f"Pothole Detector Loaded & Ran. Detections found: {len(dets)}")
    except Exception as e:
        logger.error(f"Pothole Detector Failed: {e}")

    # 2. Verify YOLO11 Road Segmenter (leeyunjai)
    logger.info("--- 2. Testing YOLO11 Road Segmenter (leeyunjai/yolo11-road-seg) ---")
    try:
        road_seg = RoadYOLO() # Default weights in code is leeyunjai
        res = road_seg.segment(dummy_frame)
        logger.success(f"Road Segmenter Loaded & Ran. Mask shape: {res.mask.shape}")
    except Exception:
        logger.error(f"Road Segmenter Failed:\n{traceback.format_exc()}")

    # 3. Verify Indian Traffic Sign Detector (akanksha-2002)
    logger.info("--- 3. Testing Indian Traffic Sign Detector ---")
    try:
        # We need to point to weights or let it download if possible
        # For verification, we'll try to load whatever is in config
        sign_det = YOLO11Detector(weights_path="models/weights/indian_signs_yolov8.pt")
        dets = sign_det.detect(dummy_frame)
        logger.success(f"Sign Detector Loaded & Ran. Detections: {len(dets)}")
    except Exception as e:
        logger.warning(f"Sign Detector skipped or failed (Likely missing local weights): {e}")

    # 4. Verify Crack Detector (U-Net + ResNet)
    logger.info("--- 4. Testing Crack Detector (U-Net + ResNet) ---")
    try:
        crack_det = CrackDetector(
            unet_weights="models/weights/crack_unet.pth",
            classifier_weights="models/weights/crack_type_resnet18.pth"
        )
        res = crack_det.detect(dummy_frame)
        logger.success(f"Crack Detector Loaded & Ran. Has crack: {res.has_crack}")
    except Exception as e:
        logger.warning(f"Crack Detector skipped or failed: {e}")

    # 5. Verify Road Surface Classifier
    logger.info("--- 5. Testing Road Surface Classifier ---")
    try:
        surf_clf = RoadSurfaceClassifier(weights_path="models/weights/surface_classifier.pth")
        surf, conf = surf_clf.classify(dummy_frame)
        logger.success(f"Surface Classifier Loaded & Ran. Result: {surf} ({conf:.2f})")
    except Exception as e:
        logger.warning(f"Surface Classifier skipped or failed: {e}")

    logger.info("✅ Verification Complete.")

if __name__ == "__main__":
    verify_all_models()
