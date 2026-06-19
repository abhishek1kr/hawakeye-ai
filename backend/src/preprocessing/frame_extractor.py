"""
Frame Extractor — Extracts frames from dashcam video at configurable intervals.
IMP-06: Uses cv2.CAP_PROP_POS_FRAMES seek instead of read-and-discard for 4-5x speedup.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple
from loguru import logger


class FrameExtractor:
    """
    Extracts frames from a video file at a configurable frame skip interval.
    Yields (frame_index, timestamp_sec, frame_bgr) tuples.

    IMP-06: Seeks directly to target frames instead of decoding every frame,
    resulting in ~4-5x faster extraction for high frame_skip values.
    """

    def __init__(self, frame_skip: int = 5, target_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            frame_skip: Process every Nth frame. Default 5 (~6 FPS from 30 FPS video).
            target_size: Optional (width, height) to resize frames. None = no resize.
        """
        self.frame_skip = frame_skip
        self.target_size = target_size

    def extract(self, video_path: str) -> Generator:
        """
        Generator that yields frames from the video using direct seeking.

        Yields:
            Tuple[int, float, np.ndarray]: (frame_index, timestamp_sec, frame_bgr)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {video_path.name} | {total_frames} frames @ {fps:.1f} FPS | frame_skip={self.frame_skip}")

        # IMP-06: Check if this codec supports seeking (some containers don't)
        supports_seek = self._check_seek_support(cap, total_frames)

        frame_idx = 0
        processed = 0

        try:
            if supports_seek and self.frame_skip > 1:
                # Fast path: seek directly to each target frame
                while frame_idx < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    timestamp = frame_idx / fps
                    if self.target_size:
                        frame = cv2.resize(frame, self.target_size)
                    yield frame_idx, timestamp, frame
                    processed += 1
                    frame_idx += self.frame_skip
            else:
                # Fallback: sequential read (needed for some compressed formats)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % self.frame_skip == 0:
                        timestamp = frame_idx / fps
                        if self.target_size:
                            frame = cv2.resize(frame, self.target_size)
                        yield frame_idx, timestamp, frame
                        processed += 1
                    frame_idx += 1
        finally:
            cap.release()

        logger.info(f"Extracted {processed} frames (seek={'yes' if supports_seek else 'no'}, every {self.frame_skip} frames)")

    def _check_seek_support(self, cap: cv2.VideoCapture, total_frames: int) -> bool:
        """
        Quick probe: seek to frame 10% into the video and verify position.
        Some container formats (e.g. MJPEG streams) don't support accurate seeking.
        """
        if total_frames < 10:
            return False
        target = total_frames // 10
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        # Allow ±2 frame tolerance
        return abs(actual - target) <= 2

    def get_video_info(self, video_path: str) -> dict:
        """Returns metadata about the video file."""
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info = {
                "fps": fps,
                "total_frames": total,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_sec": total / max(fps, 1),
            }
        finally:
            cap.release()
        return info
