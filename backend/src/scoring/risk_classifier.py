"""
Risk Classifier — Aggregates frame-level scores into segment and road-level risk.
"""
from typing import List, Dict
import numpy as np
from .safety_scorer import ScoreResult


class RiskClassifier:
    """
    Aggregates per-frame ScoreResults over a road segment or full video.
    Produces a segment risk summary with percentile breakdown.
    """

    def __init__(self, segment_length_frames: int = 50):
        self.segment_length = segment_length_frames
        self._all_scores: List[ScoreResult] = []

    def add(self, result: ScoreResult) -> None:
        self._all_scores.append(result)

    def segment_summary(self) -> Dict:
        """Break scores into segments and compute risk per segment."""
        if not self._all_scores:
            return {}
        scores = [r.score for r in self._all_scores]
        n = len(scores)
        segments = []
        for i in range(0, n, self.segment_length):
            chunk = scores[i:i + self.segment_length]
            avg = float(np.mean(chunk))
            risk = self._level(avg)
            segments.append({"start_frame": i, "end_frame": i + len(chunk),
                              "avg_score": round(avg, 1), "risk_level": risk})
        return {"segments": segments, "total_segments": len(segments)}

    def overall_summary(self) -> Dict:
        """Full video summary statistics."""
        if not self._all_scores:
            return {}
        scores = np.array([r.score for r in self._all_scores])
        return {
            "mean_score": round(float(scores.mean()), 1),
            "min_score": round(float(scores.min()), 1),
            "max_score": round(float(scores.max()), 1),
            "p10_score": round(float(np.percentile(scores, 10)), 1),
            "overall_risk": self._level(float(scores.mean())),
            "frames_good": int((scores >= 80).sum()),
            "frames_moderate": int(((scores >= 60) & (scores < 80)).sum()),
            "frames_poor": int(((scores >= 40) & (scores < 60)).sum()),
            "frames_critical": int((scores < 40).sum()),
        }

    @staticmethod
    def _level(score: float) -> str:
        if score >= 80: return "GOOD"
        if score >= 60: return "MODERATE"
        if score >= 40: return "POOR"
        return "CRITICAL"

    def reset(self) -> None:
        self._all_scores = []
