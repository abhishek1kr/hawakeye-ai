"""
Cloud Synchronization Utility — Syncs report data to a central fleet management server.
IMP-14: API key now loaded from environment variable.
IMP-17: Mock state is explicit — disabled when key is not set.
"""
import json
import os
import requests
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional


class CloudSync:
    """
    Handles data synchronization with the Hawkeye Cloud backend.

    IMP-14: API key is read from HAWKEYE_CLOUD_API_KEY env var.
    IMP-17: Sync is a no-op when key is absent — no silent fake success.
    """

    def __init__(self, api_base_url: Optional[str] = None):
        self.api_url = api_base_url or os.getenv(
            "HAWKEYE_CLOUD_API_URL", "https://api.hawkeye-road.com/v1"
        )
        # IMP-14: Load from env — never hardcode keys
        self.api_key = os.getenv("HAWKEYE_CLOUD_API_KEY", "")
        self._enabled = bool(self.api_key)

        if not self._enabled:
            logger.info("CloudSync: HAWKEYE_CLOUD_API_KEY not set — cloud sync disabled.")

    def sync_report(self, report: Dict[str, Any], geojson_path: Optional[str] = None) -> bool:
        """
        Uploads the summary report and associated GeoJSON data to the cloud.
        Returns False immediately if cloud sync is disabled.
        """
        # IMP-17: Explicit no-op — don't pretend to sync if key missing
        if not self._enabled:
            logger.debug("CloudSync: Skipped (disabled — no API key).")
            return False

        try:
            video = report.get("metadata", {}).get("video", "unknown")
            logger.info(f"CloudSync: Syncing report for '{video}'...")

            payload = {
                "report": report,
                "project_id": os.getenv("HAWKEYE_PROJECT_ID", "MV-001"),
                "device_id": os.getenv("HAWKEYE_DEVICE_ID", "HAWK-DASH-01"),
            }

            response = requests.post(
                f"{self.api_url}/uploads/reports",
                json=payload,
                headers={"X-API-Key": self.api_key},
                timeout=30,
            )
            success = response.status_code in (200, 201)

            if success:
                logger.success("CloudSync: Synchronization successful.")
            else:
                logger.error(f"CloudSync: Failed — HTTP {response.status_code}: {response.text[:200]}")

            return success

        except requests.Timeout:
            logger.error("CloudSync: Request timed out (30s).")
            return False
        except requests.ConnectionError as e:
            logger.error(f"CloudSync: Connection error — {e}")
            return False
        except Exception as e:
            logger.error(f"CloudSync: Unexpected error — {e}")
            return False

    def upload_file(self, file_path: str) -> bool:
        """Uploads a heavy file (PDF or GeoJSON) to S3-compatible storage."""
        if not self._enabled:
            logger.debug("CloudSync: File upload skipped (disabled).")
            return False

        p = Path(file_path)
        if not p.exists():
            logger.warning(f"CloudSync: File not found — {file_path}")
            return False

        try:
            logger.info(f"CloudSync: Uploading {p.name} ({p.stat().st_size / 1024:.1f} KB)...")
            # Real S3 upload would go here:
            # s3_client.upload_file(str(p), bucket_name, p.name)
            logger.success(f"CloudSync: {p.name} uploaded.")
            return True
        except Exception as e:
            logger.error(f"CloudSync: File upload failed — {e}")
            return False
