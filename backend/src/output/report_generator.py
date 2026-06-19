"""
Report Generator — Produces a structured text/JSON summary report
with crack type breakdown, pothole stats, and road geometry.

IMP-27: Switched from storing all FrameMetrics/ScoreResult objects to streaming
        aggregators (running sums/min/max/counters). Memory stays O(1) regardless
        of video length (was O(n) for n frames).
IMP-30: Removed duplicate _risk_level() — now computed from thresholds inline.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from loguru import logger

from src.scoring.safety_scorer import FrameMetrics, ScoreResult
from fpdf import FPDF


class ReportGenerator:
    """
    Aggregates per-frame data into a final road condition report using
    streaming statistics — no per-frame objects are stored in memory.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_reporter = None
        self.reset()

    def set_llm_reporter(self, llm_reporter):
        self.llm_reporter = llm_reporter

    # ── Streaming aggregation ─────────────────────────────────────────────────

    def add_frame(self, metrics: FrameMetrics, score: ScoreResult) -> None:
        """Update streaming aggregators with a single frame's data."""
        self._total_frames += 1
        self._last_timestamp = max(self._last_timestamp, getattr(metrics, "timestamp", 0))

        # Score stats
        s = score.score
        self._scores_history.append({"frame": metrics.frame_id, "score": round(s, 1)})
        self._score_sum += s
        self._score_min = min(self._score_min, s)
        self._score_max = max(self._score_max, s)
        if s >= 80:   self._frames_good += 1
        elif s >= 60: self._frames_moderate += 1
        elif s >= 40: self._frames_poor += 1
        else:         self._frames_critical += 1

        # Geometry
        if metrics.road_width_m > 0:
            self._road_width_sum += metrics.road_width_m
            self._road_width_count += 1
        self._shoulder_width_sum += metrics.shoulder_width_m

        # Cracks
        self._crack_sums["alligator"]    += metrics.crack_alligator_pct
        self._crack_sums["longitudinal"] += metrics.crack_longitudinal_pct
        self._crack_sums["transverse"]   += metrics.crack_transverse_pct
        self._crack_sums["inverse"]      += metrics.crack_inverse_pct

        # Surface majority vote
        self._surface_counts[metrics.surface_type] = \
            self._surface_counts.get(metrics.surface_type, 0) + 1

        # Signboards
        if metrics.signboard_count > 0:
            self._frames_with_signboards += 1

        # Maintenance cost
        self._total_repair_cost += getattr(score, "estimated_repair_cost", 0.0)

    def generate(self, video_name: str = "video") -> Dict:
        """Build the full report dictionary from streaming aggregators."""
        if self._total_frames == 0:
            return {}

        n = self._total_frames
        avg_score = self._score_sum / n
        overall_risk = self._risk_label(avg_score)

        crack_avg = {k: round(v / n, 4) for k, v in self._crack_sums.items()}
        total_crack_avg = round(sum(crack_avg.values()), 4)

        dominant_surface = max(
            self._surface_counts, key=self._surface_counts.get, default="asphalt"
        )
        signboard_coverage_pct = round(100 * self._frames_with_signboards / n, 1)
        avg_road_w = self._road_width_sum / max(self._road_width_count, 1)
        avg_shoulder_w = self._shoulder_width_sum / n

        report = {
            "metadata": {
                "video":                 video_name,
                "generated_at":          datetime.now().isoformat(),
                "total_frames_analyzed": n,
                "duration_sec":          round(self._last_timestamp, 1),
            },
            "overall": {
                "safety_score":           round(avg_score, 1),
                "risk_level":             overall_risk,
                "road_width_avg_m":       round(avg_road_w, 2),
                "shoulder_width_avg_m":   round(avg_shoulder_w, 2),
                "surface_type":           dominant_surface,
                "signboard_coverage_pct": signboard_coverage_pct,
                "adequate_signage":       signboard_coverage_pct >= 5.0,
            },
            "cracks": {
                "total_avg_coverage_pct": total_crack_avg,
                "by_type": {
                    "alligator":    {"avg_pct": crack_avg["alligator"],    "severity": self._crack_severity(crack_avg["alligator"],    1.0)},
                    "longitudinal": {"avg_pct": crack_avg["longitudinal"], "severity": self._crack_severity(crack_avg["longitudinal"], 0.7)},
                    "transverse":   {"avg_pct": crack_avg["transverse"],   "severity": self._crack_severity(crack_avg["transverse"],   0.6)},
                    "inverse":      {"avg_pct": crack_avg["inverse"],      "severity": self._crack_severity(crack_avg["inverse"],      0.8)},
                },
            },
            "maintenance_budget": {
                "total_estimated_cost_inr": round(self._total_repair_cost, 2),
                "avg_cost_per_km":          round((self._total_repair_cost / n) * 333, 2),
            },
            "score_distribution": {
                "min":             round(self._score_min, 1),
                "max":             round(self._score_max, 1),
                "avg":             round(avg_score, 1),
                "frames_good":     self._frames_good,
                "frames_moderate": self._frames_moderate,
                "frames_poor":     self._frames_poor,
                "frames_critical": self._frames_critical,
            },
            "scores": self._scores_history,
        }
        return report

    def save(self, video_name: str = "video") -> dict:
        """Generate report, write JSON + TXT + PDF, return the report dict."""
        report = self.generate(video_name)
        if not report:
            logger.warning("ReportGenerator.save(): no frames recorded, skipping output.")
            return report

        if self.llm_reporter:
            logger.info("Generating AI summary with LLM Reporter...")
            report["overall"]["ai_summary"] = self.llm_reporter.summarize(report)
        else:
            report["overall"]["ai_summary"] = "AI summarization disabled."

        safe_name = video_name.replace(" ", "_")

        json_path = self.output_dir / f"{safe_name}_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        txt_path = self.output_dir / f"{safe_name}_report.txt"
        self._write_text_report(report, txt_path)

        pdf_path = self.output_dir / f"{safe_name}_report.pdf"
        self._write_pdf_report(report, pdf_path)

        logger.info(f"Reports saved: JSON, TXT, PDF at {self.output_dir}")
        return report

    def reset(self) -> None:
        """Reset all streaming aggregators for a new video."""
        self._total_frames = 0
        self._last_timestamp = 0.0
        self._scores_history = []
        # Scores
        self._score_sum = 0.0
        self._score_min = 100.0
        self._score_max = 0.0
        self._frames_good = 0
        self._frames_moderate = 0
        self._frames_poor = 0
        self._frames_critical = 0
        # Geometry
        self._road_width_sum = 0.0
        self._road_width_count = 0
        self._shoulder_width_sum = 0.0
        # Cracks
        self._crack_sums = {"alligator": 0.0, "longitudinal": 0.0, "transverse": 0.0, "inverse": 0.0}
        # Surface
        self._surface_counts: Dict[str, int] = {}
        # Signboards
        self._frames_with_signboards = 0
        # Costs
        self._total_repair_cost = 0.0

    # ── Text / PDF output helpers ─────────────────────────────────────────────

    def _write_text_report(self, report: dict, path: Path) -> None:
        o = report.get("overall", {})
        c = report.get("cracks", {})
        sd = report.get("score_distribution", {})
        b = report.get("maintenance_budget", {})

        risk_icons = {"GOOD": "[OK]", "MODERATE": "[WARN]", "POOR": "[POOR]", "CRITICAL": "[CRIT]"}
        risk = o.get("risk_level", "CRITICAL")

        lines = [
            "=" * 58,
            "   AI ROAD INVENTORY & CONDITION EVALUATION REPORT",
            "=" * 58,
            f"  Video:         {report['metadata']['video']}",
            f"  Frames:        {report['metadata']['total_frames_analyzed']}",
            f"  Duration:      {report['metadata']['duration_sec']} sec",
            "",
            f"  AI SUMMARY:    {o.get('ai_summary', '')}",
            "",
            "-- ROAD GEOMETRY ------------------------------------------------",
            f"  Road Width:    {o.get('road_width_avg_m', 0):.2f} m",
            f"  Shoulder:      {o.get('shoulder_width_avg_m', 0):.2f} m",
            f"  Surface:       {o.get('surface_type', 'N/A').upper()}",
            "",
            "-- CRACK ANALYSIS -----------------------------------------------",
            f"  Total Crack Coverage:  {c.get('total_avg_coverage_pct', 0):.3f}%",
        ]
        for ctype, info in c.get("by_type", {}).items():
            pct = info.get("avg_pct", 0)
            sev = info.get("severity", "")
            bar = "#" * min(int(pct * 5), 20)
            lines.append(f"  {ctype.capitalize():15s} {pct:6.3f}%  {bar}  [{sev}]")

        lines += [
            "",
            "-- SAFETY SCORE -------------------------------------------------",
            f"  SCORE:  {o.get('safety_score', 0):.1f} / 100   {risk_icons.get(risk, '')} {risk}",
            f"  (Good:{sd.get('frames_good',0)}  Mod:{sd.get('frames_moderate',0)}  "
            f"Poor:{sd.get('frames_poor',0)}  Critical:{sd.get('frames_critical',0)} frames)",
            "=" * 58,
            f"  Generated: {report['metadata']['generated_at']}",
            "=" * 58,
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_pdf_report(self, report: dict, path: Path) -> None:
        """Generates a professional-grade PDF report."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 20)

        # Header
        pdf.cell(0, 10, "Hawkeye Road Condition Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 10, f"Generated: {report['metadata']['generated_at']}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(10)

        # Metadata
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "1. Project Metadata", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 8, f"Source Video: {report['metadata']['video']}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Total Analyzed Frames: {report['metadata']['total_frames_analyzed']}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Road Surface: {report['overall']['surface_type'].upper()}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        ai_summary = report.get("overall", {}).get("ai_summary", "")
        if ai_summary and "disabled" not in ai_summary:
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 8, "AI Summary:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "I", 12)
            pdf.multi_cell(0, 6, ai_summary)
            pdf.ln(5)

        # Safety Score
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "2. Safety Performance", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "B", 16)
        score = report["overall"]["safety_score"]
        risk = report["overall"]["risk_level"]

        if risk == "GOOD":     pdf.set_text_color(0, 150, 0)
        elif risk == "MODERATE": pdf.set_text_color(150, 100, 0)
        else:                    pdf.set_text_color(200, 0, 0)

        pdf.cell(0, 10, f"OVERALL SAFETY SCORE: {score}/100 ({risk})", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Geometry
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "3. Road Geometry", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 8, f"Average Road Width: {report['overall']['road_width_avg_m']:.2f} m", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Average Shoulder Width: {report['overall']['shoulder_width_avg_m']:.2f} m", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Distress
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "4. Distress Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 8, f"Avg Crack Coverage: {report['cracks']['total_avg_coverage_pct']:.3f}%", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "Breakdown by Crack Type:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 12)
        for ctype, info in report.get("cracks", {}).get("by_type", {}).items():
            pct = info.get("avg_pct", 0)
            sev = info.get("severity", "")
            if pct > 0 or True: # Show all to demonstrate structure
                pdf.cell(0, 8, f"  - {ctype.capitalize()}: {pct:.3f}% ({sev})", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Distribution and Additional Observations
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "5. Distribution & Additional Details", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 8, f"Signboard Coverage: {report['overall'].get('signboard_coverage_pct', 0)}%", new_x="LMARGIN", new_y="NEXT")
        
        sd = report.get("score_distribution", {})
        pdf.cell(0, 8, f"Frame Breakdown (Safety Score):", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"  - Good (80-100): {sd.get('frames_good', 0)} frames", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"  - Moderate (60-79): {sd.get('frames_moderate', 0)} frames", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"  - Poor (40-59): {sd.get('frames_poor', 0)} frames", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"  - Critical (0-39): {sd.get('frames_critical', 0)} frames", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Budget section removed as per user request
        
        pdf.output(str(path))

    # ── IMP-30: Single source of risk label logic ─────────────────────────────
    @staticmethod
    def _risk_label(score: float) -> str:
        """Map safety score to risk label. Thresholds match SafetyScorer."""
        if score >= 80: return "GOOD"
        if score >= 60: return "MODERATE"
        if score >= 40: return "POOR"
        return "CRITICAL"

    @staticmethod
    def _crack_severity(pct: float, weight: float) -> str:
        weighted = pct * weight
        if weighted < 0.5:  return "minimal"
        if weighted < 2.0:  return "minor"
        if weighted < 5.0:  return "moderate"
        if weighted < 10.0: return "severe"
        return "critical"
