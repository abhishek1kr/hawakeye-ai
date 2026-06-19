"""
Road Segmenter — Base class and result container.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SegmentationResult:
    """Container for segmentation outputs."""
    mask: np.ndarray  # 0=road, 1=shoulder, 2=background
    road_mask: np.ndarray
    shoulder_mask: np.ndarray
    
    road_width_m: float = 0.0
    shoulder_width_m: float = 0.0
    lane_count_estimate: int = 2
    road_coverage_pct: float = 0.0
    
    def __post_init__(self):
        # Calculate coverage
        total_pixels = self.mask.size
        road_pixels = np.count_nonzero(self.mask == 0)
        self.road_coverage_pct = (road_pixels / total_pixels) * 100.0


class RoadSegmenter:
    """Base class for all road segmenters."""
    
    def __init__(self, weights_path=None, device="auto", imgsz=512, pixels_per_meter=55.0):
        import torch
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.imgsz = imgsz
        self.pixels_per_meter = pixels_per_meter
        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path):
        raise NotImplementedError

    def segment(self, frame: np.ndarray) -> SegmentationResult:
        """Process a frame and return structured results."""
        mask = self._deep_segment(frame)
        
        # Split masks
        road_mask = (mask == 0).astype(np.uint8) * 255
        shoulder_mask = (mask == 1).astype(np.uint8) * 255
        
        # Calculate widths (heuristic: find median horizontal span)
        h, w = mask.shape[:2]
        road_width_px = 0
        # Check a few lines for more stability
        for y in [int(h*0.5), int(h*0.6), int(h*0.7)]:
            row = mask[y, :]
            road_width_px = max(road_width_px, np.count_nonzero(row == 0))
            
        road_width_m = road_width_px / self.pixels_per_meter
        
        # Estimate shoulder width
        shoulder_width_px = 0
        for y in [int(h*0.5), int(h*0.6), int(h*0.7)]:
            row = mask[y, :]
            shoulder_width_px = max(shoulder_width_px, np.count_nonzero(row == 1))
        shoulder_width_m = shoulder_width_px / self.pixels_per_meter

        return SegmentationResult(
            mask=mask,
            road_mask=road_mask,
            shoulder_mask=shoulder_mask,
            road_width_m=round(road_width_m, 2),
            shoulder_width_m=round(shoulder_width_m, 2),
            lane_count_estimate=max(1, int(road_width_m / 3.5))
        )

    def _deep_segment(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def draw_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Visual overlay. Resizes mask to match frame if sizes differ."""
        h, w = frame.shape[:2]
        # Defensive resize — mask from YOLO inference may be smaller than frame
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(
                mask.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST  # Nearest for label maps
            ).astype(np.uint8)

        overlay = frame.copy()
        overlay[mask == 0] = [0, 255, 0]    # Road → Green
        overlay[mask == 1] = [0, 255, 255]  # Shoulder → Yellow
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
