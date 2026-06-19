import numpy as np
import cv2
import torch
from loguru import logger
from PIL import Image

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class DepthEstimator:
    """Estimates monocular depth using Hugging Face transformers (e.g. DepthAnything)."""
    
    def __init__(self, model_name: str = "LiheYoung/depth-anything-small-hf", device: str = "auto"):
        self.model_name = model_name
        self._load_model(model_name, device)

    def _load_model(self, model_name: str, device: str) -> None:
        if pipeline is None:
            logger.error("transformers is not installed. Please install transformers.")
            self.estimator = None
            return

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cpu":
            device = "cpu"
        else:
            try:
                device = int(device) if device.isdigit() else 0
            except:
                device = "cpu"

        logger.info(f"Loading HF Depth Estimator: {model_name} on {device}...")
        try:
            self.estimator = pipeline("depth-estimation", model=model_name, device=device)
            logger.info(f"Loaded {model_name} successfully.")
        except Exception as e:
            logger.error(f"Failed to load depth estimator: {e}")
            self.estimator = None

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns a depth map as a numpy array.
        Higher values usually mean closer objects depending on the model, 
        or it's an absolute metric depth depending on the specific model.
        """
        if not self.estimator:
            return np.zeros(frame.shape[:2], dtype=np.float32)

        # Convert OpenCV BGR to RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        try:
            # The pipeline returns a dict with 'predicted_depth' (tensor) and 'depth' (PIL image)
            result = self.estimator(image)
            # We want the raw depth values for calculations
            predicted_depth = result["predicted_depth"].squeeze().cpu().numpy()
            
            # Resize back to original frame shape just in case
            depth_map = cv2.resize(predicted_depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            return depth_map
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return np.zeros(frame.shape[:2], dtype=np.float32)
