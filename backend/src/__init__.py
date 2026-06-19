"""
AI-Based Automated Road Inventory & Condition Evaluation System
Main pipeline orchestrator.
"""
from src.preprocessing import FrameExtractor, VideoStabilizer, GPSParser, ImageEnhancer, HeartbeatSync
from src.detection import YOLO11Detector, CrackDetector, RoadSurfaceClassifier
from src.segmentation import RoadSegFormer
from src.measurement import PixelToMeter, RoadGeometry
from src.tracking import DeepSort
from src.scoring import SafetyScorer, RiskClassifier
from src.output import CSVWriter, GeoJSONWriter, ReportGenerator
