import numpy as np
import cv2
import torch
from typing import List, Optional
from loguru import logger
from PIL import Image

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

from .object_detector import ObjectDetector, Detection


class HFZeroShotDetector(ObjectDetector):
    def __init__(
        self,
        weights_path: str = "google/owlvit-base-patch32",
        prompts: List[str] = None,
        confidence: float = 0.15,
        device: str = "auto",
        imgsz: int = 640
    ):
        if prompts is None:
            prompts = ["traffic sign", "guardrail"]
        self.prompts = prompts
        # Calling base class will trigger self._load_model
        super().__init__(weights_path=weights_path, confidence=confidence, iou=0.45, device=device, imgsz=imgsz)

    def _load_model(self, weights_path: str, device: str) -> None:
        if pipeline is None:
            logger.error("transformers is not installed. Please install transformers.")
            self.detector = None
            return

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cpu":
            device = "cpu"
        else:
            # e.g., '0', 'cuda:0'
            try:
                device = int(device) if device.isdigit() else 0
            except:
                device = "cpu"

        logger.info(f"Loading HF Zero-Shot Detector: {weights_path} on {device}...")
        try:
            self.detector = pipeline("zero-shot-object-detection", model=weights_path, device=device)
            logger.info(f"Loaded {weights_path} successfully.")
        except Exception as e:
            logger.error(f"Failed to load zero-shot detector: {e}")
            self.detector = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.detector or not self.prompts:
            return []

        # Convert OpenCV BGR to RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        try:
            results = self.detector(
                image,
                candidate_labels=self.prompts,
            )
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

        detections = []
        for i, res in enumerate(results):
            score = res["score"]
            if score < self.confidence:
                continue
            
            box = res["box"]
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            label = res["label"]

            detections.append(Detection(
                label=label,
                confidence=float(score),
                xmin=int(xmin),
                ymin=int(ymin),
                xmax=int(xmax),
                ymax=int(ymax),
                class_id=i,  # arbitrary index
                track_id=None
            ))

        return detections
