import logging

from typing import Optional

from backend.analysis_engine.request import AnalysisMode
from backend.analysis_engine.decision_engine import decide
from backend.api.schemas import PredictionResult

logger = logging.getLogger("crisismind")


def build_response(
    analysis_id: str,
    analysis_mode: AnalysisMode,
    prediction: PredictionResult,
    model_used: str,
    processing_time_ms: float,
    metadata: Optional[dict] = None,
) -> dict:
    decision = decide(prediction.severity)

    response = {
        "success": True,
        "analysis_id": analysis_id,
        "analysis_mode": analysis_mode.value,
        "prediction": {
            "severity": prediction.severity,
            "confidence": prediction.confidence,
        },
        "decision": {
            "priority": decision["priority"],
            "emergency_level": decision["emergency_level"],
            "infrastructure_impact": decision["infrastructure_impact"],
            "recommended_actions": decision["recommended_actions"],
        },
        "model_used": model_used,
        "model": model_used,
        "processing_time_ms": processing_time_ms,
    }
    if metadata:
        response["metadata"] = metadata

    logger.info(
        "Response built: id=%s mode=%s severity=%s confidence=%.2f model=%s time=%.1fms",
        analysis_id,
        analysis_mode.value,
        prediction.severity,
        prediction.confidence,
        model_used,
        processing_time_ms,
    )
    return response


def build_error_response(
    analysis_id: str,
    error: str,
    processing_time_ms: float,
    error_type: str = "unknown",
) -> dict:
    logger.error("Error response: id=%s type=%s error=%s time=%.1fms", analysis_id, error_type, error, processing_time_ms)
    return {
        "success": False,
        "analysis_id": analysis_id,
        "error": error,
        "error_type": error_type,
        "processing_time_ms": processing_time_ms,
    }
