import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from loguru import logger
from .road_segmenter import RoadSegmenter, SegmentationResult

class RoadSegFormer(RoadSegmenter):
    """
    SegFormer-based road segmentation model.
    Optimized for high-precision boundary detection on Indian highways.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        weights_path: Optional[str] = None,
        device: str = "auto",
        imgsz: int = 512,
        pixels_per_meter: float = 55.0,
    ):
        self.model_name = model_name
        super().__init__(weights_path, device, imgsz, pixels_per_meter)

    def _load_model(self, weights_path):
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            
            logger.info(f"Initializing SegFormer: {self.model_name}")
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            
            if weights_path and Path(weights_path).exists():
                # Load custom fine-tuned SegFormer weights if available
                model = SegformerForSemanticSegmentation.from_pretrained(weights_path)
            else:
                # Fallback to pretrained base
                model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
                logger.warning(f"Using default SegFormer weights: {self.model_name}")

            model.to(self.device)
            model.eval()
            return model
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            return None

    def _deep_segment(self, frame: np.ndarray) -> np.ndarray:
        """Run SegFormer inference."""
        # SegFormer prefers inputs in RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess using the processor
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, num_classes, H/4, W/4]

        # Upsample logits to original frame size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        
        # Get labels
        pred = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # Remap Cityscapes classes to our 3 classes (road, shoulder, background)
        # Cityscapes: 0=road, 1=sidewalk, ...
        # Our target: 0=road, 1=shoulder (sidewalk), 2=background
        target_mask = np.full(pred.shape, 2, dtype=np.uint8)
        target_mask[pred == 0] = 0  # Road
        target_mask[pred == 1] = 1  # Sidewalk -> Shoulder
        
        return target_mask
