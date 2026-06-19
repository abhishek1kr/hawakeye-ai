"""Undistort — Thin wrapper for lens distortion removal."""
import cv2
import numpy as np
from .camera_calibration import CameraCalibration


class Undistorter:
    def __init__(self, calibration: CameraCalibration):
        self.calibration = calibration
        # Pre-compute optimal maps for speed
        h = calibration.image_height
        w = calibration.image_width
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            calibration.camera_matrix,
            calibration.dist_coeffs,
            None,
            calibration.camera_matrix,
            (w, h),
            cv2.CV_16SC2,
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Fast undistortion using precomputed remap tables."""
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
