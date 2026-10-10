import io
import asyncio
import uuid
import logging
from typing import Optional, Callable

from PIL import Image

from backend.analysis_engine import AnalysisEngine, AnalysisRequest
from backend.analysis_engine.decision_engine import decide
from backend.batch_processing.progress import ProgressTracker

logger = logging.getLogger("crisismind")


class BatchProcessor:
    def __init__(self, on_file_complete: Optional[Callable] = None):
        self._batches: dict[str, ProgressTracker] = {}
        self._file_data: dict[str, dict[str, bytes]] = {}
        self._csv_map: dict[str, dict[str, str]] = {}
        self._on_file_complete = on_file_complete

    def create_batch(self, batch_id: str, files: dict[str, bytes],
                     csv_rows: Optional[list[dict]] = None) -> ProgressTracker:
        self._file_data[batch_id] = files
        if csv_rows:
            self._csv_map[batch_id] = {
                row["filename"].strip(): row for row in csv_rows
            }
        tracker = ProgressTracker(batch_id, len(files))
        self._batches[batch_id] = tracker
        return tracker

    def get_tracker(self, batch_id: str) -> Optional[ProgressTracker]:
        return self._batches.get(batch_id)

    async def process_batch(self, batch_id: str, engine: AnalysisEngine):
        tracker = self._batches.get(batch_id)
        if not tracker:
            logger.error("Batch %s not found for processing", batch_id)
            return

        files = self._file_data.get(batch_id, {})
        csv_lookup = self._csv_map.get(batch_id, {})

        tracker.start()

        for filename, contents in files.items():
            tracker.current_file = filename

            try:
                pil_image = None
                text = None

                try:
                    pil_image = Image.open(io.BytesIO(contents))
                except Exception as e:
                    pass

                csv_row = csv_lookup.get(filename)
                if csv_row and csv_row.get("text", "").strip():
                    text = csv_row["text"].strip()

                if pil_image is None and not text:
                    tracker.complete_one(filename, {
                        "success": False, "filename": filename,
                        "error": "No valid image or text provided for file",
                        "processing_time_ms": 0,
                    })
                    continue

                request = AnalysisRequest(
                    image=pil_image,
                    text=text,
                    source="batch_processing",
                )

                result = await asyncio.to_thread(engine.analyze, request)

                if result.get("success"):
                    decision = decide(result["prediction"]["severity"])
                    enriched = {
                        "success": True,
                        "filename": filename,
                        "severity": result["prediction"]["severity"],
                        "confidence": result["prediction"]["confidence"],
                        "priority": decision["priority"],
                        "infrastructure_impact": decision["infrastructure_impact"],
                        "emergency_level": decision["emergency_level"],
                        "recommended_actions": decision["recommended_actions"],
                        "processing_time_ms": result.get("processing_time_ms", 0),
                    }
                else:
                    enriched = {
                        "success": False,
                        "filename": filename,
                        "error": result.get("error", "Analysis failed"),
                        "processing_time_ms": result.get("processing_time_ms", 0),
                    }

                tracker.complete_one(filename, enriched)

                if self._on_file_complete:
                    try:
                        self._on_file_complete(enriched, batch_id)
                    except Exception:
                        logger.exception("on_file_complete callback failed for %s", filename)

            except Exception as e:
                logger.exception("Unexpected error processing %s in batch %s", filename, batch_id)
                tracker.complete_one(filename, {
                    "success": False, "filename": filename,
                    "error": str(e), "processing_time_ms": 0,
                })

                if self._on_file_complete:
                    try:
                        self._on_file_complete({
                            "success": False, "filename": filename,
                            "error": str(e), "processing_time_ms": 0,
                        }, batch_id)
                    except Exception:
                        logger.exception("on_file_complete callback failed for %s", filename)

        tracker.finish()

    def get_results(self, batch_id: str) -> Optional[list[dict]]:
        tracker = self._batches.get(batch_id)
        if not tracker:
            return None
        return tracker.results

    def cleanup(self, batch_id: str):
        self._batches.pop(batch_id, None)
        self._file_data.pop(batch_id, None)
        self._csv_map.pop(batch_id, None)
