import pandas as pd
import numpy as np
from typing import Optional, Dict
from loguru import logger
from pathlib import Path

class HeartbeatSync:
    """
    Synchronizes video frames with metadata (GPS, Chainage, INS) 
    using a 64-bit Heartbeat primary key.
    """

    def __init__(self, heartbeat_csv_path: Optional[str] = None):
        self.data = pd.DataFrame()
        if heartbeat_csv_path:
            self.load_metadata(heartbeat_csv_path)

    def load_metadata(self, path: str):
        """Load the _heartbeat.csv file containing frame metadata."""
        try:
            path = Path(path)
            if not path.exists():
                logger.error(f"Heartbeat CSV missing: {path}")
                return
            
            # Expecting columns: heartbeat, frame_id, chainage, lat, lon, alt, pitch, roll, speed
            self.data = pd.read_csv(path)
            
            # Ensure heartbeat is index for O(1) lookup
            if 'heartbeat' in self.data.columns:
                self.data.set_index('heartbeat', inplace=True)
            elif 'frame_id' in self.data.columns:
                self.data.set_index('frame_id', inplace=True)
                
            logger.info(f"Loaded {len(self.data)} metadata records from {path.name}")
        except Exception as e:
            logger.error(f"Failed to load heartbeat data: {e}")

    def get_metadata(self, heartbeat_id: int) -> Optional[Dict]:
        """Lookup metadata for a specific heartbeat/frame ID."""
        if self.data.empty or heartbeat_id not in self.data.index:
            return None
        
        row = self.data.loc[heartbeat_id]
        if isinstance(row, pd.Series):
            return row.to_dict()
        else:
            # Handle duplicate indices if any
            return row.iloc[0].to_dict()

    def estimate_chainage(self, frame_idx: int, fps: float, start_chainage: float) -> float:
        """Fallback: Estimate chainage based on frame count if metadata is missing."""
        # Simple estimation: dist = velocity * time
        # This is a placeholder; real sync uses the CSV.
        return start_chainage + (frame_idx / fps) * 0.01  # Assume 10m/s = 36km/h
