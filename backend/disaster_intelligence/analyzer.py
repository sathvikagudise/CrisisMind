"""Bridges DisasterEvent objects into the Unified Analysis Engine.

Usage::
    analyzer = DisasterAnalyzer(engine, store)
    result = analyzer.analyze(event)
"""

import logging
import time

from backend.disaster_intelligence.schemas import DisasterEvent
from backend.disaster_intelligence.store import AnalysisStore
from backend.analysis_engine import (
    AnalysisEngine,
    AnalysisRequest,
    AnalysisMode,
)

logger = logging.getLogger("crisismind.disaster_analyzer")


class DisasterAnalyzer:
    """Converts a ``DisasterEvent`` into an ``AnalysisRequest``, delegates
    to the engine, and persists the result in the store."""

    def __init__(self, engine: AnalysisEngine, store: AnalysisStore):
        self._engine = engine
        self._store = store

    def analyze(self, event: DisasterEvent) -> dict:
        start = time.perf_counter()

        self._store.update_status(event.event_id, "PROCESSING")

        text = event.title or ""
        if event.description:
            text = f"{text} {event.description}"

        request = AnalysisRequest(
            image=None,
            text=text.strip(),
            metadata={
                "disaster_type": event.disaster_type,
                "source": event.source,
                "event_id": event.event_id,
                "country": event.country,
            },
            source=event.source,
            analysis_mode=AnalysisMode.TEXT_ONLY,
        )

        engine_result = self._engine.analyze(request)
        elapsed = round((time.perf_counter() - start) * 1000, 2)

        if not engine_result.get("success"):
            self._store.update_status(event.event_id, "READY_FOR_AI")
            return {
                "success": False,
                "event": {
                    "id": event.event_id,
                    "title": event.title,
                    "type": event.disaster_type,
                    "country": event.country,
                },
                "error": engine_result.get("error", "Analysis failed"),
                "processing_time_ms": elapsed,
            }

        prediction = engine_result["prediction"]
        decision = engine_result["decision"]

        response = {
            "success": True,
            "event": {
                "id": event.event_id,
                "title": event.title,
                "type": event.disaster_type,
                "country": event.country,
            },
            "analysis": {
                "severity": prediction["severity"],
                "confidence": prediction["confidence"],
                "priority": decision["priority"],
                "infrastructure_impact": decision["infrastructure_impact"],
                "emergency_level": decision["emergency_level"],
                "recommended_actions": decision["recommended_actions"],
            },
            "model_used": engine_result.get("model_used") or engine_result.get("model") or "GatedFusion v3",
            "analysis_mode": engine_result.get("analysis_mode") or "TEXT_ONLY",
            "processing_time_ms": elapsed,
        }

        self._store.store(event.event_id, response)
        logger.info(
            "Analyzed event=%s severity=%s confidence=%.1f time=%.1fms",
            event.event_id, prediction["severity"], prediction["confidence"], elapsed,
        )
        return response
