from __future__ import annotations
import csv
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

class SingleCSVLogger:
    def __init__(self, filepath: str = "data/all_events.csv"):
        self.path = Path(filepath)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = [
            "timestamp_utc", "request_id",
            "object_ID", "request_time", "size_bytes"
        ]
        self._lock = asyncio.Lock()
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.headers).writeheader()

    async def append(self, row: Dict[str, Any]) -> None:
        async with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.headers)
                w.writerow({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "request_id": row.get("request_id"),
                    "object_ID": row.get("object_ID"),
                    "request_time": row.get("request_time"),
                    "size_bytes": row.get("size_bytes"),
                })