import sys
import os
import cv2
import numpy as np
import yaml
from loguru import logger

# Add backend dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection import HFZeroShotDetector, DepthEstimator
from src.output.llm_reporter import LLMReporter

def test_inference():
    logger.info("Starting inference test...")
    
    # Create dummy image
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Draw something so depth estimator has features
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(img, (600, 400), 100, (0, 0, 255), -1)
    
    logger.info("1. Testing HFZeroShotDetector...")
    try:
        zs = HFZeroShotDetector(
            weights_path="google/owlvit-base-patch32",
            prompts=["traffic sign"],
            confidence=0.1
        )
        dets = zs.detect(img)
        logger.info(f"ZeroShot Detections: {dets}")
    except Exception as e:
        logger.error(f"ZeroShot detector failed: {e}")
        
    logger.info("2. Testing DepthEstimator...")
    try:
        de = DepthEstimator(
            model_name="LiheYoung/depth-anything-small-hf"
        )
        depth_map = de.estimate(img)
        logger.info(f"Depth map shape: {depth_map.shape}, mean: {depth_map.mean()}")
    except Exception as e:
        logger.error(f"Depth estimator failed: {e}")

    logger.info("3. Testing LLMReporter...")
    try:
        reporter = LLMReporter(model_name="google/flan-t5-small")
        mock_report = {
            "overall": {
                "safety_score": 55.4,
                "risk_level": "POOR",
                "surface_type": "asphalt"
            },
            "cracks": {
                "total_avg_coverage_pct": 12.5
            },
            "maintenance_budget": {
                "total_estimated_cost_inr": 450000
            }
        }
        summary = reporter.summarize(mock_report)
        logger.info(f"Generated AI Summary: {summary}")
    except Exception as e:
        logger.error(f"LLM Reporter failed: {e}")

    logger.info("Tests completed.")

if __name__ == "__main__":
    test_inference()
