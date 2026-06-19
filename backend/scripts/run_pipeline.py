#!/usr/bin/env python3
"""
CLI Entry Point — Run the full road evaluation pipeline.

Usage:
  python scripts/run_pipeline.py --video path/to/video.mp4 --gps path/to/track.gpx
  python scripts/run_pipeline.py --video path/to/video.mp4
  python scripts/run_pipeline.py --video path/to/video.mp4 --output results/
"""
import sys
import click
from pathlib import Path
from loguru import logger

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


@click.command()
@click.option("--video",  "-v", required=True,  help="Path to dashcam video (.mp4/.avi)")
@click.option("--gps",    "-g", default=None,   help="Path to GPS file (.gpx or .csv)")
@click.option("--output", "-o", default="outputs", help="Output directory (default: outputs/)")
@click.option("--config", "-c", default="config/model_config.yaml", help="Model config YAML")
@click.option("--skip",   "-s", default=5, type=int, help="Frame skip interval (default: 5)")
@click.option("--no-vis", is_flag=True, help="Disable visualization frame saving")
def main(video, gps, output, config, skip, no_vis):
    """AI-Based Road Inventory & Condition Evaluation Pipeline."""
    video_path = Path(video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    logger.info(f"Starting pipeline on: {video_path.name}")
    if gps:
        logger.info(f"GPS file: {gps}")

    from src.pipeline import RoadEvaluationPipeline
    import yaml
    import os

    # Override frame_skip and visualization in config
    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("inference", {})
    cfg["inference"]["frame_skip"] = skip
    cfg["inference"]["save_visualizations"] = not no_vis

    # Write temp config
    tmp_config = Path("config/_run_config.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)

    pipeline = RoadEvaluationPipeline(config_path=str(tmp_config))
    report = pipeline.run(
        video_path=str(video_path),
        gps_path=gps,
        output_dir=output,
    )

    # Print summary
    o = report.get("overall", {})
    c = report.get("cracks", {})
    click.echo("\n" + "="*52)
    click.echo("  ROAD EVALUATION COMPLETE")
    click.echo("="*52)
    click.echo(f"  Safety Score : {o.get('safety_score', 0):.1f}/100")
    click.echo(f"  Risk Level   : {o.get('risk_level', 'N/A')}")
    click.echo(f"  Road Width   : {o.get('road_width_avg_m', 0):.2f} m")
    click.echo(f"  Surface      : {o.get('surface_type', 'N/A')}")
    click.echo(f"  Total Cracks : {c.get('total_avg_coverage_pct', 0):.3f}%")
    click.echo(f"    Alligator  : {c.get('by_type', {}).get('alligator', {}).get('avg_pct', 0):.3f}%")
    click.echo(f"    Longitudinal: {c.get('by_type', {}).get('longitudinal', {}).get('avg_pct', 0):.3f}%")
    click.echo(f"    Transverse : {c.get('by_type', {}).get('transverse', {}).get('avg_pct', 0):.3f}%")
    click.echo(f"    Inverse    : {c.get('by_type', {}).get('inverse', {}).get('avg_pct', 0):.3f}%")
    click.echo(f"  Output dir   : {output}/")
    click.echo("="*52)


if __name__ == "__main__":
    main()
