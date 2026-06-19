"""Detection package."""
from .yolo11_detector import YOLO11Detector, Detection
from .crack_detector import CrackDetector, CrackAnalysis
from .crack_type_classifier import CrackTypeClassifier, CrackTypeResult, CRACK_TYPES
from .road_surface_classifier import RoadSurfaceClassifier
from .hf_zeroshot_detector import HFZeroShotDetector
from .depth_estimator import DepthEstimator

__all__ = [
    "YOLO11Detector", "Detection",
    "CrackDetector", "CrackAnalysis",
    "CrackTypeClassifier", "CrackTypeResult", "CRACK_TYPES",
    "RoadSurfaceClassifier",
    "HFZeroShotDetector",
    "DepthEstimator",
]
