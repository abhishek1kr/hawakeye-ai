"""Video Stabilizer — Reduces camera shake using optical flow.
IMP-07: Fixed O(n²) _smooth_transforms() — now uses O(1) rolling window mean.
"""
import cv2
import numpy as np
from loguru import logger
from collections import deque


class VideoStabilizer:
    """
    Stabilizes shaky dashcam footage using Lucas-Kanade optical flow.
    Smooths camera motion with a rolling average filter.

    IMP-07: The original implementation recomputed smooth transforms for ALL
    history on every frame (O(n²)). Now uses a deque-based rolling window
    that computes only the last smoothing_radius*2 transforms — O(1) per frame.
    """

    def __init__(self, smoothing_radius: int = 15):
        self.smoothing_radius = smoothing_radius
        self._prev_gray: np.ndarray = None
        # IMP-07: Use a bounded deque instead of a growing list
        self._window: deque = deque(maxlen=smoothing_radius * 2 + 1)

    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply stabilization to a single frame incrementally."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self._prev_gray is None:
                self._prev_gray = gray
                return frame

            # Ensure same size (different resolution frames crash optical flow)
            if gray.shape != self._prev_gray.shape:
                self._prev_gray = gray
                return frame

            # Detect feature points
            prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=30,
                blockSize=3,
            )

            if prev_pts is None or len(prev_pts) < 10:
                self._prev_gray = gray
                return frame

            # Track with optical flow
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None)
            valid = status.flatten() == 1
            prev_pts_valid = prev_pts[valid]
            curr_pts_valid = curr_pts[valid]

            if len(prev_pts_valid) < 4:
                self._prev_gray = gray
                return frame

            # Estimate affine transform
            transform, _ = cv2.estimateAffinePartial2D(prev_pts_valid, curr_pts_valid)
            if transform is None:
                self._prev_gray = gray
                return frame

            dx = transform[0, 2]
            dy = transform[1, 2]
            da = np.arctan2(transform[1, 0], transform[0, 0])

            # IMP-07: Append to bounded deque — automatically drops old entries
            self._window.append([dx, dy, da])
            self._prev_gray = gray

            if len(self._window) < 2:
                return frame

            # IMP-07: O(1) rolling mean over the bounded window
            window_arr = np.array(self._window)
            smoothed_last = window_arr.mean(axis=0)
            raw_last = window_arr[-1]

            dx_corr = smoothed_last[0] - raw_last[0]
            dy_corr = smoothed_last[1] - raw_last[1]
            da_corr = smoothed_last[2] - raw_last[2]

            h, w = frame.shape[:2]
            M = np.array([
                [np.cos(da_corr), -np.sin(da_corr), dx_corr],
                [np.sin(da_corr),  np.cos(da_corr), dy_corr],
            ], dtype=np.float32)

            return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        except cv2.error as e:
            logger.warning(f"Stabilization skipped (OpenCV error): {e}")
            self._prev_gray = None
            return frame
        except Exception as e:
            logger.warning(f"Stabilization skipped: {e}")
            self._prev_gray = None
            return frame

    def reset(self) -> None:
        """Reset state for a new video."""
        self._prev_gray = None
        self._window.clear()
