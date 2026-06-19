"""
CSV Writer — Writes per-frame road condition data to CSV files.
"""
import csv
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from loguru import logger

from src.scoring.safety_scorer import FrameMetrics, ScoreResult


# Frame-level CSV columns
FRAME_COLUMNS = [
    "frame_id", "timestamp_sec",
    "lat", "lon", "speed_kmh",
    "road_width_m", "shoulder_width_m", "lane_count",
    "surface_type",
    "crack_total_pct", "crack_alligator_pct", "crack_longitudinal_pct",
    "crack_transverse_pct", "crack_inverse_pct",
    "signboard_count", "guardrail_present", "drainage_present",
    "safety_score", "risk_level",
]


class CSVWriter:
    """
    Writes per-frame metrics to a CSV file for further analysis in Excel/pandas.
    """

    def __init__(self, output_path: str = "outputs/frame_report.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._row_count = 0

    def open(self) -> None:
        """Open the CSV file and write the header row."""
        self._file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FRAME_COLUMNS)
        self._writer.writeheader()
        logger.info(f"CSV writer opened: {self.output_path}")

    def write_frame(self, metrics: FrameMetrics, score: ScoreResult) -> None:
        """Write a single frame's data as one CSV row."""
        if self._writer is None:
            self.open()
        row = {
            "frame_id": metrics.frame_id,
            "timestamp_sec": round(metrics.timestamp, 3),
            "lat": metrics.lat if metrics.lat is not None else "",
            "lon": metrics.lon if metrics.lon is not None else "",
            "speed_kmh": round(metrics.speed_kmh, 1),
            "road_width_m": round(metrics.road_width_m, 2),
            "shoulder_width_m": round(metrics.shoulder_width_m, 2),
            "lane_count": metrics.lane_count,
            "surface_type": metrics.surface_type,
            "crack_total_pct": round(metrics.crack_total_pct, 4),
            "crack_alligator_pct": round(metrics.crack_alligator_pct, 4),
            "crack_longitudinal_pct": round(metrics.crack_longitudinal_pct, 4),
            "crack_transverse_pct": round(metrics.crack_transverse_pct, 4),
            "crack_inverse_pct": round(metrics.crack_inverse_pct, 4),
            "signboard_count": metrics.signboard_count,
            "guardrail_present": int(metrics.guardrail_present),
            "drainage_present": int(metrics.drainage_present),
            "safety_score": round(score.score, 1),
            "risk_level": score.risk_level,
        }
        self._writer.writerow(row)
        self._row_count += 1

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            logger.info(f"CSV closed: {self.output_path} ({self._row_count} rows)")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
