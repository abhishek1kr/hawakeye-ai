import cv2
import numpy as np
from loguru import logger

class Dehazer:
    """
    Implements Dark Channel Prior (DCP) dehazing for hazy/foggy road footage.
    Reference: He et al., "Single Image Haze Removal Using Dark Channel Prior"
    """

    def __init__(self, window_size: int = 15, omega: float = 0.95, t_min: float = 0.1):
        self.window_size = window_size
        self.omega = omega  # Haze removal factor (0.95 is standard)
        self.t_min = t_min  # Lower bound for transmission map

    def get_dark_channel(self, img: np.ndarray) -> np.ndarray:
        """Find the dark channel of an image."""
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.window_size, self.window_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel

    def estimate_atmospheric_light(self, img: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
        """Estimate the atmospheric light (A) from the top 0.1% brightest pixels in dark channel."""
        h, w = dark_channel.shape
        num_pixels = h * w
        num_brightest = max(1, int(num_pixels * 0.001))
        
        # Flatten and get indices of top 0.1% pixels
        indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
        
        # Of these pixels, find the one with highest intensity in the original image
        brightest_pixels = img.reshape(-1, 3)[indices]
        A = np.mean(brightest_pixels, axis=0)
        return A

    def estimate_transmission(self, img: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Estimate the transmission map."""
        norm_img = img.astype(np.float32) / A
        dark_channel_norm = self.get_dark_channel(norm_img)
        transmission = 1 - self.omega * dark_channel_norm
        return transmission

    def recover_radiance(self, img: np.ndarray, A: np.ndarray, transmission: np.ndarray) -> np.ndarray:
        """Recover the final dehazed image."""
        t = cv2.max(transmission, self.t_min)
        t_broadcast = np.broadcast_to(t[:, :, np.newaxis], img.shape)
        
        res = ((img.astype(np.float32) - A) / t_broadcast) + A
        return np.clip(res, 0, 255).astype(np.uint8)

    def dehaze(self, frame: np.ndarray) -> np.ndarray:
        """Complete dehazing pipeline."""
        try:
            dark_channel = self.get_dark_channel(frame)
            A = self.estimate_atmospheric_light(frame, dark_channel)
            transmission = self.estimate_transmission(frame, A)
            
            # Refine transmission map (improves edges)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
                transmission = cv2.ximgproc.guidedFilter(guide=gray, src=transmission.astype(np.float32), 
                                                        radius=40, eps=0.001)
            else:
                # Fallback refinement using bilateral filter if ximgproc is missing
                transmission = cv2.bilateralFilter(transmission.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=5)
            
            result = self.recover_radiance(frame, A, transmission)
            return result
        except Exception as e:
            logger.error(f"Dehazing failed: {e}. Returning original frame.")
            return frame

def fast_dehaze(frame: np.ndarray) -> np.ndarray:
    """A faster, CLAHE-based approach if DCP is too slow."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
