import cv2
import numpy as np
from .dehazing import Dehazer, fast_dehaze

class ImageEnhancer:
    """
    Consolidated image enhancement pipeline for road inspection.
    Includes dehazing, color correction, and feature sharpening.
    """

    def __init__(self, use_dehazing: bool = True, sharpen_factor: float = 1.2):
        self.use_dehazing = use_dehazing
        self.dehazer = Dehazer()
        self.sharpen_factor = sharpen_factor

    def color_correct(self, img: np.ndarray) -> np.ndarray:
        """Gray World color balance to remove color casts."""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """CLAHE (Contrast Limited Adaptive Histogram Equalization) to reveal fine cracks in shadows/glare."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def sharpen(self, img: np.ndarray) -> np.ndarray:
        """Unsharp mask to enhance fine details like cracks."""
        gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
        unsharp_image = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
        return unsharp_image

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Main enhancement pipeline."""
        processed = frame.copy()

        # 1. Dehazing (Critical for highway visibility)
        if self.use_dehazing:
            processed = self.dehazer.dehaze(processed)

        # 2. Color Balance
        processed = self.color_correct(processed)

        # 3. Noise Reduction (Bilateral preserves edges)
        processed = cv2.bilateralFilter(processed, d=5, sigmaColor=75, sigmaSpace=75)

        # 4. Sharpening
        processed = self.sharpen(processed)

        # 5. CLAHE (Local Contrast Enhancement for Crack Detection)
        processed = self.apply_clahe(processed)

        return processed

def quick_enhance(frame: np.ndarray) -> np.ndarray:
    """Fast version for real-time streaming preview."""
    enhanced = fast_dehaze(frame)
    # Simple sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(enhanced, -1, kernel)
