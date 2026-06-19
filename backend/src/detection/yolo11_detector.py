import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from loguru import logger
from .object_detector import Detection, ObjectDetector

class YOLO11Detector(ObjectDetector):
    """
    YOLO11-based road object detector.
    Specialized for high-performance detection of signs and cracks.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence: float = 0.40,
        iou: float = 0.45,
        device: str = "auto",
        imgsz: int = 640,
        task: str = "detect" # "detect" or "segment"
    ):
        self.task = task
        # Override weights if YOLO11 defaults are needed
        if weights_path is None or not Path(weights_path).exists():
            # Use specific pre-trained weights from Hugging Face if local missing
            if task == "segment":
                weights_path = "leeyunjai/yolo11-road-seg" # Auto-download from HF
            else:
                weights_path = "yolo11m.pt"
            logger.info(f"Using YOLO11 {task} weights from Hugging Face: {weights_path}")
            
        super().__init__(weights_path, confidence, iou, device, imgsz)

    def _load_model(self, weights_path: str, device: str) -> None:
        try:
            from ultralytics import YOLO
            
            # If path looks like a Hugging Face repo (contains / and doesn't end in .pt)
            if "/" in weights_path and not weights_path.endswith(".pt") and not Path(weights_path).exists():
                logger.info(f"Downloading weights from Hugging Face: {weights_path}")
                from huggingface_hub import hf_hub_download
                
                repo_mapping = {
                    "athifsaleem/yolo11m-model": "yolo11m.pt",
                    "leeyunjai/yolo11-road-seg": "yolo11m-road-seg.pt",
                }
                filename = repo_mapping.get(weights_path, "best.pt") # Fallback to best.pt
                
                try:
                    weights_path = hf_hub_download(repo_id=weights_path, filename=filename)
                    logger.info(f"Downloaded HF weights to: {weights_path}")
                except Exception as e:
                    logger.error(f"Failed to download from HF: {e}")
                    # Fallback to base model if HF fails
                    weights_path = "yolo11m.pt" if self.task == "detect" else "yolo11m-seg.pt"
            
            self.model = YOLO(weights_path)
            logger.info(f"YOLO11 {self.task.upper()} loaded: {weights_path}")

            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
        except ImportError:
            logger.error("Required libraries (ultralytics, huggingface_hub) not installed.")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection/segmentation using YOLO11."""
        results = self.model(frame, conf=self.confidence, iou=self.iou, imgsz=self.imgsz, verbose=False)
        
        detections = []
        if not results:
            return detections
            
        res = results[0]
        # Process detections (boxes)
        if res.boxes:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    xmin=int(xyxy[0]),
                    ymin=int(xyxy[1]),
                    xmax=int(xyxy[2]),
                    ymax=int(xyxy[3]),
                    class_id=cls_id
                ))
        
        # If task is segment, the results will also contain masks
        self.last_results = res
        return detections

    def get_masks(self):
        """Returns segmentation masks if available."""
        if hasattr(self, 'last_results') and self.last_results.masks is not None:
            return self.last_results.masks.data.cpu().numpy()
        return None


