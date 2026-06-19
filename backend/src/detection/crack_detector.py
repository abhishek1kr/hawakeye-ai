"""
Crack Detector — Supports YOLOv8 (RDD2022) or U-Net + ResNet-18 patch classifier.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
    import segmentation_models_pytorch as smp
except ImportError:
    torch = None
    smp = None

@dataclass
class CrackAnalysis:
    """Full crack analysis result for a single frame."""
    has_crack: bool
    total_coverage_pct: float
    mask: Optional[np.ndarray] = None          # Binary mask (H x W)
    type_breakdown: Dict[str, float] = field(default_factory=dict)  # type -> coverage %
    dominant_type: Optional[str] = None
    dominant_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "has_crack": self.has_crack,
            "total_coverage_pct": round(self.total_coverage_pct, 4),
            "type_breakdown": {k: round(v, 4) for k, v in self.type_breakdown.items()},
            "dominant_type": self.dominant_type,
            "dominant_confidence": round(self.dominant_confidence, 4),
        }

class CrackDetector:
    """
    Crack detector offering two backends:
    1. YOLOv8: Single-stage bounding box detector trained on RDD2022 (fast).
    2. U-Net + ResNet-18: Pixel-level segmenter combined with type classifier (high precision).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        model_type: str = "auto",
        device: str = "auto",
        imgsz: int = 640,
        confidence: float = 0.25,
    ):
        self.imgsz = imgsz
        self.confidence = confidence
        self.model = None
        self.classifier = None
        
        # Resolve model type
        self.model_type = model_type
        if self.model_type == "auto" and weights_path:
            if Path(weights_path).suffix == ".pth" or "unet" in weights_path.lower():
                self.model_type = "unet"
            else:
                self.model_type = "yolo"
        elif self.model_type == "auto":
            self.model_type = "yolo"
            
        # Resolve device
        if device == "auto":
            if torch is not None and torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_str = "cuda"
            else:
                self.device = torch.device("cpu") if torch is not None else "cpu"
                self.device_str = "cpu"
        else:
            if torch is not None:
                self.device = torch.device(device)
            self.device_str = device

        logger.info(f"Initializing CrackDetector with backend: {self.model_type.upper()}")

        if weights_path and Path(weights_path).exists():
            if self.model_type == "yolo":
                if YOLO is None:
                    logger.warning("ultralytics not installed. Crack detection disabled.")
                    return
                try:
                    self.model = YOLO(weights_path)
                    logger.info(f"YOLO Road Damage model loaded: {weights_path}")
                except Exception as e:
                    logger.error(f"Failed to load YOLO model: {e}")
            elif self.model_type == "unet":
                if smp is None or torch is None:
                    logger.warning("torch or segmentation-models-pytorch not installed. U-Net Crack detection disabled.")
                    return
                try:
                    self.model = smp.Unet(
                        encoder_name="resnet34",
                        encoder_weights=None,
                        in_channels=3,
                        classes=1,
                        activation=None,
                    ).to(self.device)
                    
                    checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    logger.info(f"U-Net Crack Segmenter loaded: {weights_path}")
                    
                    # Initialize classifier
                    from src.detection.crack_type_classifier import CrackTypeClassifier
                    classifier_weights = "models/weights/crack_type_resnet18.pth"
                    self.classifier = CrackTypeClassifier(
                        weights_path=classifier_weights,
                        device=device,
                        confidence_threshold=0.30
                    )
                except Exception as e:
                    logger.error(f"Failed to load U-Net model/classifier: {e}")
        else:
            logger.warning(f"Crack weights not found at {weights_path}. Crack detection disabled.")

        self.class_map = {
            0: "alligator",    
            1: "transverse",   
            2: "longitudinal", 
        }

    def detect(self, frame: np.ndarray) -> CrackAnalysis:
        h, w = frame.shape[:2]
        empty_analysis = CrackAnalysis(has_crack=False, total_coverage_pct=0.0, mask=np.zeros((h, w), dtype=np.uint8))

        if self.model is None:
            return empty_analysis

        if self.model_type == "yolo":
            # Run YOLOv8 inference
            results = self.model(frame, imgsz=self.imgsz, conf=self.confidence, verbose=False, device=self.device_str)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return empty_analysis

            boxes = results[0].boxes
            mask = np.zeros((h, w), dtype=np.uint8)
            type_areas = { "longitudinal": 0.0, "transverse": 0.0, "alligator": 0.0 }
            has_crack = False
            dominant_type = None
            dominant_conf = 0.0

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if cls_id not in self.class_map:
                    continue
                    
                has_crack = True
                crack_type = self.class_map[cls_id]
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                area = (x2 - x1) * (y2 - y1)
                type_areas[crack_type] += area
                
                if conf > dominant_conf:
                    dominant_conf = conf
                    dominant_type = crack_type
                    
            if not has_crack:
                return empty_analysis

            frame_area = h * w
            total_crack_area = sum(type_areas.values())
            total_coverage_pct = min((total_crack_area / frame_area) * 100.0, 100.0)
            
            type_breakdown = {}
            for ctype, carea in type_areas.items():
                type_breakdown[ctype] = (carea / frame_area) * 100.0

            return CrackAnalysis(
                has_crack=has_crack,
                total_coverage_pct=total_coverage_pct,
                mask=mask,
                type_breakdown=type_breakdown,
                dominant_type=dominant_type,
                dominant_confidence=dominant_conf
            )

        elif self.model_type == "unet":
            # 1. Preprocess
            resized = cv2.resize(frame, (self.imgsz, self.imgsz))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (rgb - mean) / std
            
            tensor_in = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            
            # 2. Run segmentation
            with torch.no_grad():
                logits = self.model(tensor_in)
                probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            
            # 3. Threshold mask
            mask_256 = (probs > self.confidence).astype(np.uint8) * 255
            
            if np.sum(mask_256) == 0:
                return empty_analysis
                
            # 4. Resize mask back
            mask = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 5. Extract contours and classify
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            type_areas = { "longitudinal": 0.0, "transverse": 0.0, "alligator": 0.0, "inverse": 0.0 }
            has_crack = False
            dominant_type = None
            dominant_conf = 0.0
            
            patches_to_classify = []
            contour_info = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:
                    continue
                    
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Exact area computation
                c_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(c_mask, [contour], -1, 255, -1)
                pixel_count = np.sum(c_mask > 0)
                
                patch = frame[y:y+h_box, x:x+w_box]
                
                patches_to_classify.append(patch)
                contour_info.append((pixel_count, contour))
                
            if patches_to_classify and self.classifier:
                has_crack = True
                class_results = self.classifier.classify_batch(patches_to_classify)
                
                for res, (pixel_count, _) in zip(class_results, contour_info):
                    ctype = res.crack_type
                    if ctype not in type_areas:
                        type_areas[ctype] = 0.0
                    type_areas[ctype] += pixel_count
                    
                    if res.confidence > dominant_conf:
                        dominant_conf = res.confidence
                        dominant_type = ctype
            elif patches_to_classify:
                has_crack = True
                for pixel_count, contour in contour_info:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    aspect = w_box / max(h_box, 1)
                    if aspect > 2.0:
                        ctype = "transverse"
                    elif aspect < 0.5:
                        ctype = "longitudinal"
                    else:
                        ctype = "alligator"
                    type_areas[ctype] += pixel_count
                    if dominant_type is None:
                        dominant_type = ctype
                        dominant_conf = 0.5
            
            if not has_crack:
                return empty_analysis
                
            frame_area = h * w
            total_crack_area = sum(type_areas.values())
            total_coverage_pct = min((total_crack_area / frame_area) * 100.0, 100.0)
            
            type_breakdown = {}
            for ctype, carea in type_areas.items():
                type_breakdown[ctype] = (carea / frame_area) * 100.0
                
            return CrackAnalysis(
                has_crack=has_crack,
                total_coverage_pct=total_coverage_pct,
                mask=mask,
                type_breakdown=type_breakdown,
                dominant_type=dominant_type,
                dominant_confidence=dominant_conf
            )

        return empty_analysis

    def draw_mask(self, frame: np.ndarray, analysis: CrackAnalysis) -> np.ndarray:
        """Overlay crack mask on frame."""
        if analysis.mask is None or not analysis.has_crack:
            return frame
        overlay = frame.copy()
        color = (0, 0, 200)  # Default red
        overlay[analysis.mask > 0] = color
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
