"""Pixel-to-meter converter using camera geometry."""
import numpy as np
from src.preprocessing.camera_calibration import CameraCalibration


class PixelToMeter:
    def __init__(self, calibration: CameraCalibration):
        self.cal = calibration

    def convert_width(self, pixel_width: float, bbox_bottom_y: float) -> float:
        dist = self.cal.estimate_distance(bbox_bottom_y)
        return self.cal.pixel_to_meter(pixel_width, dist)

    def convert_height(self, pixel_height: float, bbox_bottom_y: float) -> float:
        dist = self.cal.estimate_distance(bbox_bottom_y)
        fy = self.cal.camera_matrix[1, 1]
        return (pixel_height * dist) / fy

    def bbox_to_meters(self, bbox_xyxy, frame_height: int):
        x1, y1, x2, y2 = bbox_xyxy
        w_m = self.convert_width(x2 - x1, y2)
        h_m = self.convert_height(y2 - y1, y2)
        return w_m, h_m
