import cv2
import numpy as np
import torch
from typing import Optional
from src.detection.yolo11_detector import YOLO11Detector
from src.segmentation.road_segmenter import RoadSegmenter, SegmentationResult

class RoadYOLO(RoadSegmenter):
    """
    Road Segmentation wrapper using YOLO11-Segmentation models.
    """
    def __init__(
        self,
        weights_path: str = "leeyunjai/yolo11-road-seg",
        pixels_per_meter: float = 100.0,
        imgsz: int = 640
    ):
        # We don't call super().__init__ directly to avoid the NotImplementedError in
        # RoadSegmenter._load_model(), so we initialize all needed attributes manually.
        # BUG-07 FIX: Explicitly set self.device so inherited methods don't fail.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pixels_per_meter = pixels_per_meter
        self.imgsz = imgsz
        self.detector = YOLO11Detector(
            weights_path=weights_path,
            task="segment",
            imgsz=imgsz
        )

    def _load_model(self, weights_path):
        # Already handled by YOLO11Detector
        return None

    def _deep_segment(self, frame: np.ndarray) -> np.ndarray:
        # Run YOLO11 segment
        _ = self.detector.detect(frame)
        masks = self.detector.get_masks()

        h, w = frame.shape[:2]

        if masks is None or len(masks) == 0:
            return np.zeros((h, w), dtype=np.uint8) + 2  # Default to background

        # Combine all instance masks (max-pool over all detections)
        combined_mask = np.max(masks, axis=0)  # Shape: (inference_h, inference_w) — NOT frame size

        # ── CRITICAL FIX: Resize mask from inference size to original frame size ──
        # YOLO returns masks at imgsz resolution (e.g. 384x640), not 1080x1920.
        # Boolean indexing with mismatched shapes crashes with:
        # "boolean index did not match indexed array along axis 0"
        if combined_mask.shape != (h, w):
            combined_mask = cv2.resize(
                combined_mask.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )

        # Remap: binary mask (road=1) → RoadSegmenter format (0=road, 2=background)
        final_mask = np.full((h, w), 2, dtype=np.uint8)  # Background
        final_mask[combined_mask > 0.5] = 0              # Road
        return final_mask

    def segment(self, frame: np.ndarray) -> SegmentationResult:
        # Use the base class logic for width/lane estimation
        return super().segment(frame)
