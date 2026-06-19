"""
DeepSORT-lite — Simple Kalman filter + IoU tracker for road objects.
Used to deduplicate pothole/crack detections across frames.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class Track:
    """Single tracked object."""
    track_id: int
    class_name: str
    bbox_xyxy: List[float]
    confidence: float
    age: int = 0                  # Frames since first seen
    miss_count: int = 0           # Consecutive frames without match
    total_hits: int = 1

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class DeepSort:
    """
    Lightweight IoU-based multi-object tracker.
    Assigns consistent track IDs to road damage detections across frames.
    Prevents counting the same pothole multiple times in consecutive frames.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list) -> List[Track]:
        """
        Update tracker with new detections from the current frame.

        Args:
            detections: List of Detection objects.


        Returns:
            List of active Track objects (only confirmed tracks, age >= min_hits).
        """
        # Age all existing tracks
        for t in self._tracks.values():
            t.miss_count += 1

        if not detections:
            self._remove_dead_tracks()
            return self._active_tracks()

        det_boxes = [[d.xmin, d.ymin, d.xmax, d.ymax] for d in detections]

        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].bbox_xyxy for tid in track_ids]

        # Greedy IoU matching
        matched_dets = set()
        matched_tracks = set()

        if track_boxes:
            iou_matrix = self._iou_matrix(track_boxes, det_boxes)
            for t_idx, d_idx in np.argwhere(iou_matrix > self.iou_threshold):
                # BUG-09 FIX: Cast np.int64 indices to plain int for safe set/list ops
                t_idx, d_idx = int(t_idx), int(d_idx)
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                tid = track_ids[t_idx]
                self._tracks[tid].bbox_xyxy = det_boxes[d_idx]
                self._tracks[tid].confidence = detections[d_idx].confidence
                self._tracks[tid].miss_count = 0
                self._tracks[tid].total_hits += 1
                self._tracks[tid].age += 1
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                detections[d_idx].track_id = tid

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = Track(
                    track_id=self._next_id,
                    class_name=det.label,
                    bbox_xyxy=[det.xmin, det.ymin, det.xmax, det.ymax],
                    confidence=det.confidence,
                )

                self._tracks[self._next_id] = new_track
                det.track_id = self._next_id
                self._next_id += 1

        self._remove_dead_tracks()
        return self._active_tracks()

    def _remove_dead_tracks(self) -> None:
        dead = [tid for tid, t in self._tracks.items() if t.miss_count > self.max_age]
        for tid in dead:
            del self._tracks[tid]

    def _active_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.total_hits >= self.min_hits]

    def _iou_matrix(self, boxes_a: list, boxes_b: list) -> np.ndarray:
        matrix = np.zeros((len(boxes_a), len(boxes_b)))
        for i, a in enumerate(boxes_a):
            for j, b in enumerate(boxes_b):
                matrix[i, j] = self._iou(a, b)
        return matrix

    @staticmethod
    def _iou(a: list, b: list) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-8)

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 1

    @property
    def active_count(self) -> int:
        return len(self._active_tracks())
