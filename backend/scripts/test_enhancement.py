import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.enhancement import ImageEnhancer

def test_pipeline():
    # Create a synthetic hazy image for testing
    img = np.full((480, 640, 3), 150, dtype=np.uint8)
    cv2.circle(img, (320, 240), 100, (0, 0, 255), -1) # Red circle
    cv2.putText(img, "TEST FRAME", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Add fake haze (lighten the image)
    white_img = np.full((480, 640, 3), 255, dtype=np.uint8)
    hazy_img = cv2.addWeighted(img, 0.5, white_img, 0.5, 0)
    
    print("Testing ImageEnhancer...")
    enhancer = ImageEnhancer(use_dehazing=True)
    result = enhancer.enhance(img)
    
    print(f"Original shape: {img.shape}, Enhanced shape: {result.shape}")
    if result.shape == img.shape:
        print("Success: Enhancement pipeline executed without errors.")
    else:
        print("Error: Shape mismatch in enhancement output.")

if __name__ == "__main__":
    test_pipeline()
