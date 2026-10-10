"""In-memory store for analysis results."""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("crisismind.store")


class AnalysisStore:
    """In-memory store for disaster event analysis results.

    Tracks the lifecycle status of each event and caches its latest
    analysis output.  No database dependency.
    """

    def __init__(self):
        self._analyses: dict[str, dict] = {}
        self._statuses: dict[str, str] = {}

    # ── Write ────────────────────────────────────────────────────────────

    def store(self, event_id: str, result: dict) -> None:
        self._analyses[event_id] = {
            "result": result,
            "timestamp": datetime.utcnow(),
        }
        self._statuses[event_id] = "ANALYZED"
        logger.info("Stored analysis for event=%s", event_id)

    def update_status(self, event_id: str, status: str) -> None:
        self._statuses[event_id] = status
        logger.debug("Status updated: event=%s status=%s", event_id, status)

    # ── Read ─────────────────────────────────────────────────────────────

    def get(self, event_id: str) -> Optional[dict]:
        entry = self._analyses.get(event_id)
        if entry is not None:
            logger.debug("Analysis cache HIT event=%s", event_id)
            return entry["result"]
        logger.debug("Analysis cache MISS event=%s", event_id)
        return None

    def get_status(self, event_id: str) -> str:
        return self._statuses.get(event_id, "READY_FOR_AI")

    def get_analyzed_ids(self) -> set[str]:
        return {
            eid for eid, s in self._statuses.items() if s == "ANALYZED"
        }

    def count_analyzed(self) -> int:
        return len(self.get_analyzed_ids())

    def get_entry_timestamp(self, event_id: str):
        """Return the stored timestamp for an analyzed event."""
        entry = self._analyses.get(event_id)
        if entry is not None:
            return entry["timestamp"]
        return None

    # ── Admin ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._analyses.clear()
        self._statuses.clear()
        logger.info("Analysis store cleared")
