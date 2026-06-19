"""
GPS Parser — Parses GPS tracks from GPX, NMEA, or CSV and syncs with video timestamps.
"""
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timezone

import gpxpy
import gpxpy.gpx
from loguru import logger


@dataclass
class GPSPoint:
    """Single GPS measurement."""
    timestamp: float          # Seconds since video start (synced)
    lat: float                # Latitude (decimal degrees)
    lon: float                # Longitude (decimal degrees)
    alt: float = 0.0          # Altitude (meters)
    speed_kmh: float = 0.0    # Speed (km/h)
    heading: float = 0.0      # Heading (degrees, 0=North)

    def as_geojson_coords(self) -> List[float]:
        return [self.lon, self.lat, self.alt]


class GPSParser:
    """
    Parses GPS data from GPX/NMEA/CSV and synchronizes with video frame timestamps.
    Uses linear interpolation to estimate GPS position at any video timestamp.
    """

    def __init__(self, sync_tolerance_sec: float = 1.0):
        self.sync_tolerance_sec = sync_tolerance_sec
        self._points: List[GPSPoint] = []

    def load_gpx(self, gpx_path: str) -> None:
        """Load GPS track from GPX file."""
        gpx_path = Path(gpx_path)
        if not gpx_path.exists():
            raise FileNotFoundError(f"GPX file not found: {gpx_path}")

        with open(gpx_path) as f:
            gpx = gpxpy.parse(f)

        raw_points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for pt in segment.points:
                    raw_points.append({
                        "time": pt.time.timestamp() if pt.time else 0,
                        "lat": pt.latitude,
                        "lon": pt.longitude,
                        "alt": pt.elevation or 0.0,
                    })

        self._normalize_and_store(raw_points)
        logger.info(f"Loaded {len(self._points)} GPS points from {gpx_path.name}")

    def load_csv(self, csv_path: str, timestamp_col: str = "timestamp",
                 lat_col: str = "lat", lon_col: str = "lon",
                 alt_col: str = "alt", speed_col: str = "speed_kmh") -> None:
        """Load GPS track from CSV file."""
        csv_path = Path(csv_path)
        raw_points = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    raw_points.append({
                        "time": float(row.get(timestamp_col, 0)),
                        "lat": float(row.get(lat_col, 0)),
                        "lon": float(row.get(lon_col, 0)),
                        "alt": float(row.get(alt_col, 0)),
                        "speed": float(row.get(speed_col, 0)),
                    })
                except (ValueError, KeyError):
                    continue

        self._normalize_and_store(raw_points)
        logger.info(f"Loaded {len(self._points)} GPS points from {csv_path.name}")

    def _normalize_and_store(self, raw_points: list) -> None:
        """Normalize timestamps to start at 0.0 and compute speeds."""
        if not raw_points:
            return

        t0 = raw_points[0]["time"]
        points = []
        for i, p in enumerate(raw_points):
            speed = p.get("speed", 0.0)
            heading = 0.0
            if i > 0:
                prev = raw_points[i - 1]
                heading = self._compute_bearing(prev["lat"], prev["lon"], p["lat"], p["lon"])
                if speed == 0.0:
                    dt = p["time"] - prev["time"]
                    if dt > 0:
                        dist = self._haversine(prev["lat"], prev["lon"], p["lat"], p["lon"])
                        speed = (dist / dt) * 3.6  # m/s → km/h

            points.append(GPSPoint(
                timestamp=p["time"] - t0,
                lat=p["lat"],
                lon=p["lon"],
                alt=p["alt"],
                speed_kmh=speed,
                heading=heading,
            ))
        self._points = points

    def get_at(self, video_timestamp_sec: float) -> Optional[GPSPoint]:
        """
        Get interpolated GPS position at a given video timestamp (seconds).
        Returns None if no GPS data is loaded or timestamp is out of range.
        """
        if not self._points:
            return None

        # Find surrounding points
        for i in range(len(self._points) - 1):
            t0 = self._points[i].timestamp
            t1 = self._points[i + 1].timestamp
            if t0 <= video_timestamp_sec <= t1:
                alpha = (video_timestamp_sec - t0) / max(t1 - t0, 1e-9)
                return self._interpolate(self._points[i], self._points[i + 1], alpha)

        # Out of range — return nearest endpoint
        if video_timestamp_sec < self._points[0].timestamp:
            return self._points[0]
        return self._points[-1]

    def _interpolate(self, a: GPSPoint, b: GPSPoint, alpha: float) -> GPSPoint:
        def lerp(x, y): return x + alpha * (y - x)
        return GPSPoint(
            timestamp=lerp(a.timestamp, b.timestamp),
            lat=lerp(a.lat, b.lat),
            lon=lerp(a.lon, b.lon),
            alt=lerp(a.alt, b.alt),
            speed_kmh=lerp(a.speed_kmh, b.speed_kmh),
            heading=lerp(a.heading, b.heading),
        )

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        """Distance in meters between two lat/lon points."""
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _compute_bearing(lat1, lon1, lat2, lon2) -> float:
        """Compute initial bearing (degrees) from point 1 to point 2."""
        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    @property
    def points(self) -> List[GPSPoint]:
        return self._points

    def __len__(self) -> int:
        return len(self._points)
