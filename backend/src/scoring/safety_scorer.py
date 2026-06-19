"""
Safety Scorer — Weighted penalty system, 0-100 score.
Reads weights from config/scoring_weights.yaml.
Crack-type-aware: alligator carries highest penalty.
"""
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class FrameMetrics:
    """All measurable parameters for a single frame."""
    frame_id: int
    timestamp: float

    # Geometry
    road_width_m: float = 0.0
    shoulder_width_m: float = 0.0
    lane_count: int = 2
    chainage_m: Optional[float] = None

    # Cracks (coverage %)
    crack_total_pct: float = 0.0
    crack_alligator_pct: float = 0.0
    crack_longitudinal_pct: float = 0.0
    crack_transverse_pct: float = 0.0
    crack_inverse_pct: float = 0.0
    crack_depth_variance: float = 0.0

    # Infrastructure
    signboard_count: int = 0
    guardrail_present: bool = False
    drainage_present: bool = False
    pothole_count: int = 0

    # Surface
    surface_type: str = "asphalt"

    # GPS
    lat: Optional[float] = None
    lon: Optional[float] = None
    speed_kmh: float = 0.0


@dataclass
class ScoreResult:
    """Safety score and penalty breakdown for one frame."""
    score: float
    risk_level: str
    risk_color: str
    penalties: Dict[str, float] = field(default_factory=dict)
    estimated_repair_cost: float = 0.0
    metrics: Optional[FrameMetrics] = None

    def to_dict(self) -> dict:
        return {
            "safety_score": round(self.score, 1),
            "risk_level": self.risk_level,
            "penalties": {k: round(v, 2) for k, v in self.penalties.items()},
            "estimated_repair_cost": round(self.estimated_repair_cost, 2),
        }


class SafetyScorer:
    """
    Computes a safety score (0–100) from FrameMetrics using configurable weights.
    100 = perfect road, 0 = extremely dangerous.
    """

    def __init__(self, config_path: str = "config/scoring_weights.yaml"):
        self.cfg = self._load_config(config_path)

    def _load_config(self, path: str) -> dict:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
        logger.warning("Scoring config not found. Using built-in defaults.")
        return self._defaults()

    def score(self, m: FrameMetrics) -> ScoreResult:
        """Compute safety score for a single frame's metrics."""
        penalties = {}
        total_penalty = 0.0

        # ── Road Geometry ────────────────────────────────────────────────────
        rw_cfg = self.cfg.get("road_width", {})
        if m.road_width_m < rw_cfg.get("min_acceptable_m", 5.5):
            p = rw_cfg.get("penalty_very_narrow", 30) if m.road_width_m < 3.5 \
                else rw_cfg.get("penalty_narrow", 20)
            penalties["narrow_road"] = p
            total_penalty += p

        sw_cfg = self.cfg.get("shoulder_width", {})
        if m.shoulder_width_m < sw_cfg.get("min_acceptable_m", 0.5):
            p = sw_cfg.get("penalty_no_shoulder", 10)
            penalties["no_shoulder"] = p
            total_penalty += p

        # ── Crack Penalties (type-specific) ──────────────────────────────────
        crack_coverages = {
            "alligator":    m.crack_alligator_pct,
            "longitudinal": m.crack_longitudinal_pct,
            "transverse":   m.crack_transverse_pct,
            "inverse":      m.crack_inverse_pct,
        }
        crack_cfg = self.cfg.get("cracks", {})
        for crack_type, coverage in crack_coverages.items():
            if coverage <= 0:
                continue
            type_cfg = crack_cfg.get(crack_type, {})
            thresholds = type_cfg.get("thresholds", [])
            penalty = self._threshold_penalty(coverage, thresholds)
            if penalty > 0:
                penalties[f"crack_{crack_type}"] = penalty
                total_penalty += penalty

        # Additional penalty for deep/rough cracks based on monocular depth variance
        if m.crack_depth_variance > 5.0:
            depth_penalty = min(20.0, (m.crack_depth_variance - 5.0) * 2.0)
            penalties["crack_depth_severity"] = depth_penalty
            total_penalty += depth_penalty

        # Pothole penalty
        if m.pothole_count > 0:
            pothole_penalty = min(30.0, m.pothole_count * 15.0)  # 15 per pothole, max 30
            penalties["potholes"] = pothole_penalty
            total_penalty += pothole_penalty

        # ── Infrastructure ────────────────────────────────────────────────────
        # Per-frame signboard penalty intentionally removed:
        # A signboard is visible for only ~2-3 s per minute of road footage,
        # so penalising every zero-signboard frame drags scores down by ~5pts
        # on 95% of frames — semantically wrong. Route-level check is in
        # ReportGenerator.generate() instead.

        # ── Surface Type Modifier ─────────────────────────────────────────────
        surf_cfg = self.cfg.get("surface_type", {})
        modifier = surf_cfg.get(m.surface_type, {}).get("modifier", 0)
        if modifier != 0:
            penalties[f"surface_{m.surface_type}"] = abs(modifier)
            total_penalty += abs(modifier)

        # ── Repair Cost Estimation ────────────────────────────────────────────
        cost_cfg = self.cfg.get("repair_costs", {})
        repair_cost = 0.0
        
        # Crack costs (assume 1 frame covers ~10sqm of road surface)
        # crack_total_pct * 10sqm * cost_per_sqm
        crack_costs = cost_cfg.get("cracks", {})
        sqm_per_frame = 10.0 # Heuristic for 3.5m lane x 3m depth
        cracked_area = (m.crack_total_pct / 100.0) * sqm_per_frame
        repair_cost += cracked_area * crack_costs.get("sealing_sqm", 185)

        score = max(0.0, min(100.0, 100.0 - total_penalty))
        risk_level, risk_color = self._get_risk_level(score)

        return ScoreResult(
            score=round(score, 1),
            risk_level=risk_level,
            risk_color=risk_color,
            penalties=penalties,
            estimated_repair_cost=repair_cost,
            metrics=m,
        )

    def _threshold_penalty(self, value: float, thresholds: list) -> float:
        """Return highest applicable penalty from threshold list."""
        penalty = 0.0
        for t in thresholds:
            if isinstance(t, dict):
                if value >= t.get("coverage_pct", 999):
                    penalty = t.get("penalty", 0)
        return penalty

    def _get_risk_level(self, score: float):
        # BUG-08 FIX: Use explicit descending-threshold check so adding/reordering
        # risk levels in the YAML can never accidentally return the wrong level.
        risk_cfg = self.cfg.get("risk_levels", {})
        # Sort levels from highest min_score to lowest so first match wins
        ordered = sorted(
            risk_cfg.items(),
            key=lambda kv: kv[1].get("min_score", 0),
            reverse=True
        )
        for level, cfg in ordered:
            if score >= cfg.get("min_score", 0):
                return cfg.get("label", level.upper()), cfg.get("color", "#888888")
        return "CRITICAL", "#7c3aed"

    def _defaults(self) -> dict:
        return {
            "road_width": {"min_acceptable_m": 5.5, "penalty_narrow": 20, "penalty_very_narrow": 30},
            "shoulder_width": {"min_acceptable_m": 0.5, "penalty_no_shoulder": 10},
            "cracks": {
                "alligator":    {"thresholds": [{"coverage_pct": 1.0, "penalty": 15}, {"coverage_pct": 5.0, "penalty": 25}]},
                "longitudinal": {"thresholds": [{"coverage_pct": 2.0, "penalty": 8},  {"coverage_pct": 5.0, "penalty": 15}]},
                "transverse":   {"thresholds": [{"coverage_pct": 2.0, "penalty": 6},  {"coverage_pct": 5.0, "penalty": 12}]},
                "inverse":      {"thresholds": [{"coverage_pct": 2.0, "penalty": 10}, {"coverage_pct": 5.0, "penalty": 18}]},
            },
            "infrastructure": {"no_signboards": {"penalty": 5}},
            "surface_type": {"asphalt": {"modifier": 0}, "concrete": {"modifier": 0},
                             "gravel": {"modifier": -5}, "unpaved": {"modifier": -15}},
            "risk_levels": {
                "good":     {"min_score": 80, "label": "GOOD",     "color": "#22c55e"},
                "moderate": {"min_score": 60, "label": "MODERATE", "color": "#f59e0b"},
                "poor":     {"min_score": 40, "label": "POOR",     "color": "#ef4444"},
                "critical": {"min_score": 0,  "label": "CRITICAL", "color": "#7c3aed"},
            },
            "repair_costs": {
                "cracks": {
                    "sealing_sqm": 250
                }
            }
        }
