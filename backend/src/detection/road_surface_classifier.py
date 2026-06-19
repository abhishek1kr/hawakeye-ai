"""
Road Surface Classifier — ResNet-18 classifying asphalt/concrete/gravel/unpaved.
Dataset: RSCD (Road Surface Recognition Dataset)
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

SURFACE_CLASSES = {0: "asphalt", 1: "concrete", 2: "gravel", 3: "unpaved"}
SURFACE_COLORS = {
    "asphalt":  (100, 100, 100),
    "concrete": (180, 180, 200),
    "gravel":   (80,  140, 200),
    "unpaved":  (60,  120, 80),
}


class RoadSurfaceClassifier:
    """
    Classifies the type of road surface visible in the lower-center crop of each frame.
    Uses a ResNet-18 backbone fine-tuned on RSCD dataset.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "auto",
        imgsz: int = 224,
        confidence_threshold: float = 0.50,
    ):
        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device)
        self.model = self._load_model(weights_path)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self, weights_path: Optional[str]) -> nn.Module:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(SURFACE_CLASSES))

        if weights_path and Path(weights_path).exists():
            # BUG-15 FIX: Explicit weights_only to silence PyTorch 2.x FutureWarning
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            logger.info(f"Surface classifier loaded: {weights_path}")
        else:
            logger.warning("Surface classifier weights not found. Returning 'asphalt' default.")

        model.to(self.device)
        model.eval()
        return model

    def classify(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Classify road surface type from a frame.
        Uses the lower-center crop (road is most visible there).

        Returns:
            (surface_type, confidence): e.g. ("asphalt", 0.91)
        """
        crop = self._extract_road_crop(frame)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        class_id = probs.argmax().item()
        confidence = probs[class_id].item()
        surface = SURFACE_CLASSES[class_id]

        if confidence < self.confidence_threshold:
            surface = "asphalt"  # Default to asphalt on low confidence

        return surface, float(confidence)

    def _extract_road_crop(self, frame: np.ndarray) -> np.ndarray:
        """Extract lower 40% center 60% of frame where road surface is most visible."""
        h, w = frame.shape[:2]
        y1 = int(h * 0.60)
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)
        crop = frame[y1:h, x1:x2]
        if crop.size == 0:
            return frame
        return crop
