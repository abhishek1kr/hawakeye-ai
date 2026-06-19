"""
GeoJSON Writer — Writes GPS-tagged detections and road condition segments.
Each detection is a GeoJSON Feature with lat/lon coordinates.
"""
import json
import math
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger

from src.scoring.safety_scorer import ScoreResult, FrameMetrics
from src.preprocessing.gps_parser import GPSPoint


class GeoJSONWriter:
    """
    Writes road condition data as GeoJSON FeatureCollections.
    Supports:
    - Individual detection features (points)
    - Road segment linestrings with aggregated scores
    - Folium HTML map generation
    """

    def __init__(self, output_dir: str = "outputs/geojson"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._features: List[dict] = []
        self._track_points: List[list] = []  # [lon, lat] pairs for road track

    def add_frame(
        self,
        metrics: FrameMetrics,
        score_result: ScoreResult,
        gps_point: Optional[GPSPoint] = None,
    ) -> None:
        """Add a frame's road condition data as a GeoJSON feature."""
        if gps_point is None:
            # Provide synthetic GPS data for demo purposes so map works
            lon = 78.9629 + (metrics.frame_id * 0.00005)
            lat = 20.5937 + (metrics.frame_id * 0.00005)
            speed_kmh = 40.0
        else:
            lon, lat = gps_point.lon, gps_point.lat
            speed_kmh = gps_point.speed_kmh

        self._track_points.append([lon, lat])

        # Aggregate detections into the feature properties
        properties = {
            "frame_id": metrics.frame_id,
            "timestamp": round(metrics.timestamp, 2),
            "lat": lat,
            "lon": lon,
            "speed_kmh": round(speed_kmh, 1),
            "safety_score": score_result.score,
            "risk_level": score_result.risk_level,
            "road_width_m": metrics.road_width_m,
            "shoulder_width_m": metrics.shoulder_width_m,
            "surface_type": metrics.surface_type,
            # Cracks
            "crack_total_pct": round(metrics.crack_total_pct, 3),
            "crack_alligator_pct": round(metrics.crack_alligator_pct, 3),
            "crack_longitudinal_pct": round(metrics.crack_longitudinal_pct, 3),
            "crack_transverse_pct": round(metrics.crack_transverse_pct, 3),
            "crack_inverse_pct": round(metrics.crack_inverse_pct, 3),
            # Marker color
            "marker_color": score_result.risk_color,
        }

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat, 0]},
            "properties": properties,
        }
        self._features.append(feature)

    def add_detection_point(
        self,
        detection_type: str,
        crack_type: Optional[str],
        severity: str,
        confidence: float,
        gps_point: GPSPoint,
        extra: Optional[Dict] = None,
    ) -> None:
        """Add a single detection (crack zone) as a GeoJSON point feature."""
        properties = {
            "detection_type": detection_type,
            "crack_type": crack_type,
            "severity": severity,
            "confidence": round(confidence, 3),
            "lat": gps_point.lat,
            "lon": gps_point.lon,
            **(extra or {}),
        }
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [gps_point.lon, gps_point.lat]},
            "properties": properties,
        }
        self._features.append(feature)

    def save(self, filename: str = "road_condition.geojson") -> Path:
        """Save all collected features as a GeoJSON file."""
        collection = {
            "type": "FeatureCollection",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_features": len(self._features),
                "track_points": len(self._track_points),
            },
            "features": self._features,
        }
        if self._track_points:
            track_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": self._track_points,
                },
                "properties": {"type": "gps_track", "description": "Dashcam GPS track"},
            }
            collection["features"].insert(0, track_feature)

        out_path = self.output_dir / filename
        with open(out_path, "w") as f:
            json.dump(collection, f, indent=2)
        logger.info(f"GeoJSON saved: {out_path} ({len(self._features)} features)")
        return out_path

    def generate_map(self, filename: str = "road_condition_map.html") -> Optional[Path]:
        """Generate an interactive Folium HTML map."""
        try:
            import folium
        except ImportError:
            logger.warning("folium not installed. Skipping map generation.")
            return None

        if not self._track_points:
            logger.warning("No GPS track to map.")
            return None

        # Center map on midpoint of track
        mid = self._track_points[len(self._track_points) // 2]
        m = folium.Map(location=[mid[1], mid[0]], zoom_start=15,
                       tiles="CartoDB positron")

        # Draw GPS track
        folium.PolyLine(
            [[p[1], p[0]] for p in self._track_points],
            color="#3b82f6", weight=3, opacity=0.7,
        ).add_to(m)

        # Add feature markers
        COLOR_MAP = {
            "GOOD": "green", "MODERATE": "orange",
            "POOR": "red", "CRITICAL": "purple",
        }
        for feat in self._features:
            geom = feat["geometry"]
            if geom["type"] != "Point":
                continue
            lon, lat = geom["coordinates"][:2]
            props = feat["properties"]
            risk = props.get("risk_level", "GOOD")
            score = props.get("safety_score", 100)

            popup_html = f"""
            <b>Frame:</b> {props.get('frame_id', '?')}<br>
            <b>Score:</b> {score}/100 — {risk}<br>
            <b>Road width:</b> {props.get('road_width_m', 0):.1f}m<br>
            <b>Crack:</b> {props.get('crack_total_pct', 0):.2f}%<br>
            <b>Surface:</b> {props.get('surface_type', 'asphalt')}
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=COLOR_MAP.get(risk, "gray"),
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)

        out_path = self.output_dir / filename
        m.save(str(out_path))
        logger.info(f"Map saved: {out_path}")
        return out_path

    def reset(self) -> None:
        """Clear all features for a new video."""
        self._features = []
        self._track_points = []
