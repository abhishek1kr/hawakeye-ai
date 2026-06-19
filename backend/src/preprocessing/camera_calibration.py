"""
Camera Calibration — Loads camera intrinsics/extrinsics and applies undistortion.
"""
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger


class CameraCalibration:
    """
    Manages camera intrinsic and extrinsic parameters.
    Supports lens undistortion and Bird's Eye View (BEV) homography.
    """

    def __init__(self, config_path: str = "config/camera_params.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.camera_matrix = self._build_camera_matrix()
        self.dist_coeffs = self._build_dist_coeffs()
        self.homography_matrix = self._build_homography()
        self.pixels_per_meter = self.config["homography"].get("pixels_per_meter", 55.0)
        
        # Expose important params
        ext = self.config.get("extrinsics", {})
        self.pitch_deg = ext.get("pitch_deg", -5.0)
        
        ref = self.config.get("reference", {})
        self.lane_width_m = ref.get("lane_width_m", 3.5)
        self.shoulder_width_m = ref.get("shoulder_width_m", 1.5)

        
        logger.info(f"Camera calibration loaded: fx={self.camera_matrix[0,0]:.1f}")

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}. Using defaults.")
            return self._default_config()
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict:
        return {
            "intrinsics": {"fx": 1050.0, "fy": 1050.0, "cx": 960.0, "cy": 540.0},
            "distortion": {"k1": -0.28, "k2": 0.07, "p1": 0.0, "p2": 0.0, "k3": 0.0},
            "extrinsics": {"camera_height_m": 1.3, "pitch_deg": -5.0},
            "resolution": {"width": 1920, "height": 1080},
            "homography": {
                "src_points": [[576,810],[1344,810],[1728,1080],[192,1080]],
                "dst_points": [[0.0,10.0],[7.0,10.0],[7.0,0.0],[0.0,0.0]],
                "pixels_per_meter": 55.0,
            },
        }

    def _build_camera_matrix(self) -> np.ndarray:
        i = self.config["intrinsics"]
        return np.array([
            [i["fx"],    0.0, i["cx"]],
            [0.0,    i["fy"], i["cy"]],
            [0.0,        0.0,     1.0],
        ], dtype=np.float64)

    def _build_dist_coeffs(self) -> np.ndarray:
        d = self.config["distortion"]
        return np.array([d["k1"], d["k2"], d["p1"], d["p2"], d["k3"]], dtype=np.float64)

    def _build_homography(self) -> np.ndarray:
        h = self.config["homography"]
        src = np.array(h["src_points"], dtype=np.float32)
        dst = np.array(h["dst_points"], dtype=np.float32)
        # Scale dst to pixel space using ppm
        ppm = h.get("pixels_per_meter", 55.0)
        dst_px = dst.copy()
        dst_px[:, 0] *= ppm
        dst_px[:, 1] *= ppm
        M, _ = cv2.findHomography(src, dst_px)
        return M

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Remove lens distortion from a frame."""
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def pixel_to_meter(self, px_width: float, distance_m: float) -> float:
        """Convert pixel width to real-world meters using pinhole model."""
        fx = self.camera_matrix[0, 0]
        return (px_width * distance_m) / fx

    def estimate_distance(self, bbox_bottom_y: float) -> float:
        """
        Estimate distance to object using its bottom-edge Y position in the frame.
        Higher in the frame (small Y) = farther away.
        """
        cam_height = self.config["extrinsics"]["camera_height_m"]
        pitch_rad = np.radians(abs(self.config["extrinsics"]["pitch_deg"]))
        fy = self.camera_matrix[1, 1]
        cy = self.camera_matrix[1, 2]

        angle_from_center = np.arctan((bbox_bottom_y - cy) / fy)
        depression_angle = pitch_rad + angle_from_center
        if depression_angle <= 0:
            return 50.0  # Default far distance
        return cam_height / np.tan(depression_angle)

    @property
    def image_width(self) -> int:
        return self.config["resolution"]["width"]

    @property
    def image_height(self) -> int:
        return self.config["resolution"]["height"]

    @property
    def camera_height_m(self) -> float:
        return self.config["extrinsics"]["camera_height_m"]
