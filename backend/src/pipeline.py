"""
Main Pipeline — Orchestrates all modules to process a dashcam video.
Usage: python scripts/run_pipeline.py --video <path> --gps <path>
"""
import cv2
import yaml
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

from src.preprocessing import FrameExtractor, VideoStabilizer, CameraCalibration, GPSParser, Undistorter, ImageEnhancer, HeartbeatSync
from src.detection import CrackDetector, RoadSurfaceClassifier, YOLO11Detector, HFZeroShotDetector, DepthEstimator
from src.detection.object_detector import Detection
from src.segmentation import RoadYOLO
from src.measurement import RoadGeometry
from src.tracking import DeepSort
from src.scoring import SafetyScorer, RiskClassifier, FrameMetrics
from src.output import CSVWriter, GeoJSONWriter, ReportGenerator
from src.utils.cloud_sync import CloudSync


class RoadEvaluationPipeline:
    """
    End-to-end pipeline:
    video (+ GPS) → detections → measurements → scoring → CSV/GeoJSON/Report
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        inf = self.cfg.get("inference", {})
        self.frame_skip = inf.get("frame_skip", 5)
        self.save_vis = inf.get("save_visualizations", True)
        self.vis_every = inf.get("visualization_every_n", 30)

        logger.info("Initializing pipeline components...")
        self._init_models()
        logger.info("Pipeline ready.")

    def _init_models(self):
        pre_cfg  = self.cfg.get("pre_processing", {})
        y11_cfg  = self.cfg.get("yolo11_detector", {})
        crack_cfg = self.cfg.get("crack_detector_yolo", {})
        rsy_cfg   = self.cfg.get("road_segmenter_yolo", {})
        surf_cfg  = self.cfg.get("surface_classifier", {})
        sign_cfg  = self.cfg.get("sign_detector", {})
        zeroshot_cfg = self.cfg.get("zeroshot_detector", {})
        depth_cfg = self.cfg.get("depth_estimator", {})

        self.calibration    = CameraCalibration("config/camera_params.yaml")
        self.extractor      = FrameExtractor(frame_skip=self.frame_skip)
        self.stabilizer     = VideoStabilizer()
        self.undistorter    = Undistorter(self.calibration)
        self.gps_parser     = GPSParser()
        self.enhancer       = ImageEnhancer(
            use_dehazing=pre_cfg.get("use_dehazing", True),
            sharpen_factor=pre_cfg.get("sharpen_factor", 1.2)
        )
        self.sync           = HeartbeatSync()

        logger.info("Initializing Hawkeye AI stack (YOLO11m + SegFormer)...")
        
        self.object_detector = YOLO11Detector(
            weights_path=y11_cfg.get("weights"),
            confidence=y11_cfg.get("confidence", 0.40),
            imgsz=y11_cfg.get("imgsz", 640),
            device=y11_cfg.get("device", "auto"),
            task=y11_cfg.get("task", "detect"),
        )
        
        pothole_cfg = self.cfg.get("pothole_detector", {})
        if pothole_cfg and pothole_cfg.get("weights") and Path(pothole_cfg.get("weights")).exists():
            self.pothole_detector = YOLO11Detector(
                weights_path=pothole_cfg.get("weights"),
                confidence=pothole_cfg.get("confidence", 0.20),
                imgsz=pothole_cfg.get("imgsz", 640),
                device=pothole_cfg.get("device", "auto"),
                task="detect" # Standard detection
            )
        else:
            self.pothole_detector = None
        
        if sign_cfg and sign_cfg.get("weights") and Path(sign_cfg.get("weights")).exists():
            self.sign_detector = YOLO11Detector(
                weights_path=sign_cfg.get("weights"),
                confidence=sign_cfg.get("confidence", 0.35),
                imgsz=sign_cfg.get("imgsz", 640),
                device=sign_cfg.get("device", "auto"),
            )
        else:
            self.sign_detector = None

        if zeroshot_cfg.get("enabled", False):
            self.zeroshot_detector = HFZeroShotDetector(
                weights_path=zeroshot_cfg.get("model", "google/owlvit-base-patch32"),
                prompts=zeroshot_cfg.get("prompts", ["traffic sign", "guardrail"]),
                confidence=zeroshot_cfg.get("confidence", 0.15),
                device=zeroshot_cfg.get("device", "auto"),
            )
        else:
            self.zeroshot_detector = None

        if depth_cfg.get("enabled", False):
            self.depth_estimator = DepthEstimator(
                model_name=depth_cfg.get("model", "LiheYoung/depth-anything-small-hf"),
                device=depth_cfg.get("device", "auto")
            )
        else:
            self.depth_estimator = None

        
        self.crack_detector = CrackDetector(
            weights_path=crack_cfg.get("weights", "models/weights/road_damage_yolov8.pt"),
            device=crack_cfg.get("device", "auto"),
            imgsz=crack_cfg.get("imgsz", 640),
            confidence=crack_cfg.get("confidence", 0.25),
        )
        
        

        self.road_segmenter = RoadYOLO(
            weights_path=rsy_cfg.get("weights"),
            pixels_per_meter=self.calibration.pixels_per_meter,
            imgsz=rsy_cfg.get("imgsz", 640),
        )
        
        self.surface_classifier = RoadSurfaceClassifier(
            weights_path=surf_cfg.get("weights"),
        )

        self.road_geometry = RoadGeometry(
            calibration=self.calibration
        )
        
        self.tracker        = DeepSort()
        self.scorer         = SafetyScorer()
        self.risk_classifier = RiskClassifier()

        self.csv_w          = CSVWriter()
        self.geojson_w      = GeoJSONWriter()
        
        reporter_cfg = self.cfg.get("report_generator", {})
        if reporter_cfg.get("use_llm_summary", False):
            from src.output.llm_reporter import LLMReporter
            self.llm_reporter = LLMReporter(
                model_name=reporter_cfg.get("model", "google/flan-t5-small"),
                device=reporter_cfg.get("device", "auto")
            )
        else:
            self.llm_reporter = None
            
        self.report_gen     = ReportGenerator()
        if self.llm_reporter:
            self.report_gen.set_llm_reporter(self.llm_reporter)
            
        self.cloud_sync     = CloudSync()

    def run(
        self,
        video_path: str,
        gps_path: Optional[str] = None,
        output_dir: str = "outputs",
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """
        Process a single dashcam video end-to-end.

        Args:
            video_path: Path to .mp4 / .avi dashcam video
            gps_path:   Path to .gpx or .csv GPS file (optional)
            output_dir: Directory for CSV / GeoJSON / report outputs
            progress_callback: Callback function(current_frame, total_frames, img_vis)

        Returns:
            Summary report dict
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = output_dir / "frames"
        vis_dir.mkdir(exist_ok=True)

        video_name = video_path.stem
        logger.info(f"Processing: {video_name}")
        v_info = self.extractor.get_video_info(str(video_path))
        total_frames = v_info.get("total_frames", 0)

        # Load GPS
        if gps_path and Path(gps_path).exists():
            ext = Path(gps_path).suffix.lower()
            if ext == ".gpx":
                self.gps_parser.load_gpx(gps_path)
            elif ext == ".csv":
                if "_heartbeat" in str(gps_path):
                    self.sync.load_metadata(gps_path)
                else:
                    self.gps_parser.load_csv(gps_path)
            logger.info("GPS/Heartbeat loaded.")

        # Initialize outputs
        reporter  = self.report_gen
        reporter.output_dir = output_dir
        self.geojson_w.output_dir = output_dir
        geojson_w = self.geojson_w


        csv_path  = str(output_dir / "analysis_results.csv")

        self.tracker.reset()
        self.risk_classifier.reset()
        reporter.reset()
        self._last_surface = "asphalt"  # Reset per-video surface state

        with CSVWriter(csv_path) as csv_w:
            for frame_idx, timestamp, frame in self.extractor.extract(str(video_path)):

                # 1. Preprocess
                frame = self.stabilizer.stabilize_frame(frame)
                frame = self.undistorter.undistort(frame)
                frame = self.enhancer.enhance(frame)

                # 2. Metadata lookup (Heartbeat / GPS)
                hb_meta = self.sync.get_metadata(frame_idx)
                gps_pt = self.gps_parser.get_at(timestamp)
                
                # Update pitch if available from INS
                chainage = hb_meta.get("chainage") if hb_meta else None
                
                # If no GPS/Heartbeat is loaded, simulate chainage and GPS for UI visualization
                if chainage is None:
                    # Assume 30 km/h (8.33 m/s) -> chainage in km
                    chainage = (timestamp * 8.33) / 1000.0
                
                if gps_pt is None:
                    # Simulate a dummy GPS point starting near New Delhi for visualization
                    from src.preprocessing.gps_parser import GPSPoint
                    # Roughly 111,000 meters per degree of latitude
                    sim_lat = 28.6139 + ((timestamp * 8.33) / 111000.0)
                    sim_lon = 77.2090
                    gps_pt = GPSPoint(timestamp=timestamp, lat=sim_lat, lon=sim_lon, alt=210.0, speed_kmh=30.0)

                # 3. Object detection
                detections = self.object_detector.detect(frame)
                sign_detections = []
                if self.sign_detector:
                    sign_detections.extend(self.sign_detector.detect(frame))
                if getattr(self, "zeroshot_detector", None):
                    sign_detections.extend(self.zeroshot_detector.detect(frame))
                
                pothole_detections = []
                if getattr(self, "pothole_detector", None):
                    # Change label to 'pothole' so UI shows it clearly
                    raw_potholes = self.pothole_detector.detect(frame)
                    for d in raw_potholes:
                        d.label = "pothole"
                    pothole_detections.extend(raw_potholes)
                
                # Combine detections for tracking
                self.tracker.update(detections + sign_detections + pothole_detections)



                # 4. Crack analysis
                crack_analysis = self.crack_detector.detect(frame)

                # 4.1 Depth Estimation (if cracks present)
                crack_depth_var = 0.0
                if getattr(self, "depth_estimator", None) and crack_analysis.has_crack and crack_analysis.mask is not None:
                    depth_map = self.depth_estimator.estimate(frame)
                    mask_bin = crack_analysis.mask > 0
                    if np.any(mask_bin):
                        crack_depths = depth_map[mask_bin]
                        crack_depth_var = float(np.std(crack_depths))

                # 7. Road segmentation
                seg = self.road_segmenter.segment(frame)
                
                # 7.1 Refine measurement with RoadGeometry (BEV)
                geo_metrics = self.road_geometry.measure(seg)

                # 8. Surface classification (every 10th *processed* frame, i.e. every frame_skip*10 source frames)
                # NOTE: frame_idx here is the absolute source-video frame number, not a sequential counter,
                # so this correctly fires approx every 50 source frames when frame_skip=5.
                if frame_idx % (self.frame_skip * 10) == 0:
                    surface, _ = self.surface_classifier.classify(frame)
                    self._last_surface = surface
                surface = getattr(self, "_last_surface", "asphalt")

                # 9. Assemble metrics
                metrics = FrameMetrics(
                    frame_id=frame_idx,
                    timestamp=timestamp,
                    road_width_m=geo_metrics["road_width_m"],
                    shoulder_width_m=geo_metrics["shoulder_width_m"],
                    lane_count=geo_metrics["lane_count"],
                    chainage_m=chainage,
                    crack_total_pct=crack_analysis.total_coverage_pct,
                    crack_alligator_pct=crack_analysis.type_breakdown.get("alligator", 0),
                    crack_longitudinal_pct=crack_analysis.type_breakdown.get("longitudinal", 0),
                    crack_transverse_pct=crack_analysis.type_breakdown.get("transverse", 0),
                    crack_inverse_pct=crack_analysis.type_breakdown.get("inverse", 0),
                    crack_depth_variance=crack_depth_var,
                    pothole_count=len(pothole_detections),
                    signboard_count=len(self.object_detector.get_signboards(detections + sign_detections)),
                    surface_type=surface,
                    lat=gps_pt.lat if gps_pt else None,
                    lon=gps_pt.lon if gps_pt else None,
                    speed_kmh=gps_pt.speed_kmh if gps_pt else 0.0,
                )

                # 10. Score
                score = self.scorer.score(metrics)
                self.risk_classifier.add(score)

                # 11. Write outputs
                csv_w.write_frame(metrics, score)
                geojson_w.add_frame(metrics, score, gps_pt)
                reporter.add_frame(metrics, score)

                # 12. Save visualization & send progress
                is_vis_step = (frame_idx // self.frame_skip) % self.vis_every == 0
                is_live_step = (frame_idx // self.frame_skip) % 3 == 0  # Update live preview every 3 processed frames to save time
                
                vis_frame = None
                all_detections = detections + sign_detections + pothole_detections
                if is_vis_step or is_live_step:
                    vis_frame = self._visualize(frame, all_detections, crack_analysis, seg, score)
                
                # Only save to disk periodically to save space/I/O
                if is_vis_step and self.save_vis and vis_frame is not None:
                    cv2.imwrite(str(vis_dir / f"frame_{frame_idx:06d}.jpg"), vis_frame)

                if progress_callback:
                    progress_callback(frame_idx, total_frames, vis_frame, score, metrics)

        # Save outputs
        geojson_w.save("analysis_track.geojson")
        geojson_w.generate_map("interactive_map.html")

        report = reporter.save(video_name)  # Uses video_name for proper file naming


        # 13. Sync to Cloud — only if explicitly enabled in config + API key is set
        # IMP-20 FIX: default was True (bug), now False to match model_config.yaml
        if self.cfg.get("inference", {}).get("auto_sync_cloud", False):
            self.cloud_sync.sync_report(report)

        return report

    def _visualize(self, frame, detections, crack_analysis, seg, score) -> "np.ndarray":
        vis = self.object_detector.draw_detections(frame, detections)
        vis = self.road_segmenter.draw_mask(vis, seg.mask)
        if crack_analysis.has_crack and crack_analysis.mask is not None:
            vis = self.crack_detector.draw_mask(vis, crack_analysis)

        # Score overlay
        risk_colors = {"GOOD": (0,200,0), "MODERATE": (0,165,255),
                       "POOR": (0,80,255), "CRITICAL": (0,0,200)}
        color = risk_colors.get(score.risk_level, (128,128,128))
        label = f"Score: {score.score:.0f}/100  [{score.risk_level}]"
        cv2.rectangle(vis, (0, 0), (320, 36), (0, 0, 0), -1)
        cv2.putText(vis, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return vis
