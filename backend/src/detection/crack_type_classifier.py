"""
Crack Type Classifier — Classifies crack regions into 4 types.
Types: alligator | longitudinal | inverse | transverse
Dataset: RDD2022 (D10, D20, D30, D40)
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


# ── RDD2022 class mapping ──────────────────────────────────────────────────────
CRACK_TYPES = {
    0: "alligator",    # D10 — fatigue / interconnected mesh
    1: "longitudinal", # D20 — parallel to road axis
    2: "inverse",      # D30 — block/grid pattern (CRRI nomenclature)
    3: "transverse",   # D40 — perpendicular to road axis
}

CRACK_SEVERITY_WEIGHTS = {
    "alligator":    1.0,   # Most severe — structural failure indicator
    "longitudinal": 0.7,
    "inverse":      0.8,
    "transverse":   0.6,
}

CRACK_COLORS_BGR = {
    "alligator":    (0,   0,   220),  # Red
    "longitudinal": (0,   165, 255),  # Orange
    "inverse":      (147, 20,  255),  # Purple
    "transverse":   (0,   215, 255),  # Yellow
}


@dataclass
class CrackTypeResult:
    crack_type: str           # alligator | longitudinal | inverse | transverse
    confidence: float         # 0.0 – 1.0
    severity_weight: float    # For scoring
    class_id: int
    color_bgr: Tuple[int, int, int]

    def to_dict(self) -> dict:
        return {
            "crack_type": self.crack_type,
            "confidence": round(self.confidence, 4),
            "severity_weight": self.severity_weight,
        }


class CrackTypeClassifier:
    """
    Classifies crack image patches into one of 4 types.
    Uses a ResNet-18 backbone fine-tuned on RDD2022 (D10/D20/D30/D40).

    Falls back to rule-based classification if weights are not available.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.50,
        imgsz: int = 224,
    ):
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.device = self._resolve_device(device)
        self.model = None

        # Image normalization (ImageNet stats)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if weights_path and Path(weights_path).exists():
            self._load_model(weights_path)
        else:
            logger.warning(
                "Crack type classifier weights not found. "
                "Using rule-based fallback. Train with: "
                "models/training/train_crack_type_classifier.py"
            )

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self, weights_path: str) -> None:
        """Load fine-tuned ResNet-18 from checkpoint."""
        import torchvision.models as models
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CRACK_TYPES))

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        logger.info(f"Crack type classifier loaded: {weights_path}")

        # IMP-19: Warmup — first inference always has CUDA kernel init overhead
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.classify(dummy)
            logger.debug("Crack type classifier warmed up.")
        except Exception:
            pass

    def classify(self, crack_patch: np.ndarray) -> CrackTypeResult:
        """
        Classify a single crack image patch.

        Args:
            crack_patch: BGR image crop of a crack region (any size)

        Returns:
            CrackTypeResult with type, confidence, and severity weight
        """
        if crack_patch is None or crack_patch.size == 0:
            return self._default_result()

        if self.model is not None:
            return self._classify_deep(crack_patch)
        else:
            return self._classify_rule_based(crack_patch)

    def classify_batch(self, patches: List[np.ndarray]) -> List[CrackTypeResult]:
        """
        IMP-18: True GPU batch inference — stacks all patches into one tensor
        and runs a single forward pass. ~5-10x faster than sequential classify().
        Falls back to sequential for rule-based mode.
        """
        if not patches:
            return []

        # Filter out empty patches
        valid_patches = [(i, p) for i, p in enumerate(patches) if p is not None and p.size > 0]
        results = [self._default_result()] * len(patches)

        if not valid_patches:
            return results

        if self.model is None:
            # Rule-based: sequential (no GPU batching possible)
            for i, p in valid_patches:
                results[i] = self._classify_rule_based(p)
            return results

        try:
            # Deep batch path
            indices, batch_patches = zip(*valid_patches)
            tensors = []
            for p in batch_patches:
                rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
                tensors.append(self.transform(rgb))

            batch = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                logits = self.model(batch)
                probs_all = torch.softmax(logits, dim=1)

            for j, idx in enumerate(indices):
                probs = probs_all[j]
                class_id = probs.argmax().item()
                confidence = probs[class_id].item()
                crack_type = CRACK_TYPES[class_id]
                results[idx] = CrackTypeResult(
                    crack_type=crack_type,
                    confidence=confidence,
                    severity_weight=CRACK_SEVERITY_WEIGHTS[crack_type],
                    class_id=class_id,
                    color_bgr=CRACK_COLORS_BGR[crack_type],
                )
        except Exception as e:
            logger.warning(f"Batch classify failed, falling back to sequential: {e}")
            for i, p in valid_patches:
                results[i] = self._classify_deep(p)

        return results

    def _classify_deep(self, patch: np.ndarray) -> CrackTypeResult:
        """Deep learning classification."""
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        class_id = probs.argmax().item()
        confidence = probs[class_id].item()
        crack_type = CRACK_TYPES[class_id]

        return CrackTypeResult(
            crack_type=crack_type,
            confidence=confidence,
            severity_weight=CRACK_SEVERITY_WEIGHTS[crack_type],
            class_id=class_id,
            color_bgr=CRACK_COLORS_BGR[crack_type],
        )

    def _classify_rule_based(self, patch: np.ndarray) -> CrackTypeResult:
        """
        Rule-based fallback classifier using image analysis.
        Analyzes crack orientation using gradient direction statistics.
        """
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sobel gradients
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.degrees(np.arctan2(np.abs(gy), np.abs(gx) + 1e-8))

        # Threshold to keep strong edges
        mask = mag > (mag.mean() + mag.std())
        if mask.sum() == 0:
            return self._default_result()

        angles = angle[mask]
        h_ratio = (angles < 30).mean()   # horizontal → transverse
        v_ratio = (angles > 60).mean()   # vertical   → longitudinal
        # Connectivity analysis to distinguish alligator vs inverse
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        num_labels, _ = cv2.connectedComponents(dilated)
        fragmentation = num_labels / max(patch.shape[0] * patch.shape[1] / 100, 1)

        if h_ratio > 0.5:
            crack_type, class_id = "transverse", 3
        elif v_ratio > 0.5:
            crack_type, class_id = "longitudinal", 1
        elif fragmentation > 2.0:
            crack_type, class_id = "alligator", 0   # Highly fragmented → alligator
        else:
            crack_type, class_id = "inverse", 2     # Grid-like → inverse

        return CrackTypeResult(
            crack_type=crack_type,
            confidence=0.55,  # Low confidence for rule-based
            severity_weight=CRACK_SEVERITY_WEIGHTS[crack_type],
            class_id=class_id,
            color_bgr=CRACK_COLORS_BGR[crack_type],
        )

    def _default_result(self) -> CrackTypeResult:
        return CrackTypeResult(
            crack_type="longitudinal",
            confidence=0.0,
            severity_weight=CRACK_SEVERITY_WEIGHTS["longitudinal"],
            class_id=1,
            color_bgr=CRACK_COLORS_BGR["longitudinal"],
        )
