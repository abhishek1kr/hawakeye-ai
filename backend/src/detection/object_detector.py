import numpy as np
import cv2
from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger

@dataclass
class Detection:
    label: str
    confidence: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    class_id: int
    track_id: Optional[int] = None

class ObjectDetector(ABC):
    """Base class for all object detectors in Hawkeye."""
    def __init__(
        self,
        weights_path: str,
        confidence: float = 0.40,
        iou: float = 0.45,
        device: str = "auto",
        imgsz: int = 640
    ):
        self.weights_path = weights_path
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self._load_model(weights_path, device)

    @abstractmethod
    def _load_model(self, weights_path: str, device: str) -> None:
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        pass

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        vis = frame.copy()
        for det in detections:
            cv2.rectangle(vis, (det.xmin, det.ymin), (det.xmax, det.ymax), (0, 255, 0), 2)
            label = f"{det.label}: {det.confidence:.2f}"
            cv2.putText(vis, label, (det.xmin, det.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return vis

    def get_signboards(self, detections: List[Detection]) -> List[Detection]:
        """Returns sign detections. Includes Ryukijano model's BROKEN_SIGNAGE and FADED_SIGNAGE."""
        sign_labels = ["sign", "signboard", "traffic sign", "speed limit", "stop", "give way", "no entry", "billboard"]
        return [d for d in detections if any(s in d.label.lower() for s in sign_labels)]
