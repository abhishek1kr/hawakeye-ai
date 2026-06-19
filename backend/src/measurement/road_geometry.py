import cv2
import numpy as np
from src.segmentation.road_segmenter import SegmentationResult
from src.preprocessing.camera_calibration import CameraCalibration

class RoadGeometry:
    """
    Derives real-world road geometry using Bird's Eye View (BEV) transformation.
    """

    def __init__(self, calibration: CameraCalibration):
        self.calibration = calibration
        self.ppm = calibration.pixels_per_meter
        self.M = calibration.homography_matrix
        
        # Calculate output BEV size
        # We assume a 10m x 20m patch for the top-down view
        self.bev_w = int(10 * self.ppm)
        self.bev_h = int(20 * self.ppm)

    def to_bev(self, img: np.ndarray) -> np.ndarray:
        """Apply Bird's Eye View transformation."""
        if self.M is None:
            return img
        # Ensure we don't crash if img is small or wrong shape
        h, w = img.shape[:2]
        # Warp using the calibrated homography matrix
        return cv2.warpPerspective(img, self.M, (self.bev_w, self.bev_h))

    def measure(self, seg: SegmentationResult) -> dict:
        """
        Refine measurements from SegmentationResult using true BEV.
        """
        road_width = seg.road_width_m 
        shoulder_width = seg.shoulder_width_m

        if hasattr(seg, 'road_mask') and seg.road_mask is not None and self.M is not None:
            try:
                bev_mask = self.to_bev(seg.road_mask)
                h, w = bev_mask.shape[:2]
                widths_px = []
                
                # Sample bottom half of BEV (closer to camera = more accurate)
                for y in range(int(h * 0.5), int(h * 0.95), int(h * 0.05)):
                    row = bev_mask[y, :]
                    widths_px.append(np.count_nonzero(row > 127))
                    
                if widths_px and max(widths_px) > 0:
                    median_px = np.median([w for w in widths_px if w > 0])
                    road_width = median_px / self.ppm
            except Exception:
                pass # Fallback to original

        if hasattr(seg, 'shoulder_mask') and seg.shoulder_mask is not None and self.M is not None:
            try:
                bev_s_mask = self.to_bev(seg.shoulder_mask)
                s_widths_px = []
                h = bev_s_mask.shape[0]
                for y in range(int(h * 0.5), int(h * 0.95), int(h * 0.05)):
                    row = bev_s_mask[y, :]
                    s_widths_px.append(np.count_nonzero(row > 127))
                if s_widths_px and max(s_widths_px) > 0:
                    median_s_px = np.median([w for w in s_widths_px if w > 0])
                    shoulder_width = median_s_px / self.ppm
            except Exception:
                pass

        # Smooth output bounding limits (roads are typically 3m - 15m)
        road_width = np.clip(road_width, 2.0, 20.0)

        return {
            "road_width_m": round(road_width, 2),
            "shoulder_width_m": round(shoulder_width, 2),
            "total_width_m": round(road_width + shoulder_width, 2),
            "lane_count": max(1, int(road_width / 3.5)),
            "road_coverage_pct": seg.road_coverage_pct,
            "resolution_mm_px": round(1000.0 / self.ppm, 2),
            "dynamic_pitch_detected": False
        }
