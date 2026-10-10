"""Shared report builder — normalises any analysis result into a common dict."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.dashboard.store import AnalysisRecord

logger = logging.getLogger("crisismind.reporting")

# Characters outside the Latin-1 range that Helvetica cannot render,
# plus C1 control chars (U+007F-U+009F) which have no cp1252 mapping and
# crash fpdf's font encoding (often introduced by latin-1 mojibake upstream).
_UNICODE_RE = re.compile(r"[^\t\n\r\x20-\x7E\xA0-\xFF]")


def _sanitize(text: str | None) -> str:
    """Strip characters not supported by fpdf2 default Helvetica font."""
    if not text:
        return ""
    return _UNICODE_RE.sub("", text).strip()


def build_report_data(
    record: AnalysisRecord,
) -> dict:
    """Build a normalised report dict from any AnalysisRecord.

    The output dict contains all fields needed by the PDF and CSV generators.
    """
    now = datetime.utcnow()
    rd = record.raw_data or {}

    # Extract fields with fallbacks
    severity = record.severity
    confidence = record.confidence
    priority = record.priority
    analysis_mode = record.analysis_mode or rd.get("analysis_mode", "N/A")
    disaster_type = record.disaster_type or rd.get("disaster_type", "N/A")
    country = record.country or rd.get("country", "N/A")
    source = record.source or rd.get("source", "N/A")
    model_used = record.model_used or rd.get("model_used") or rd.get("model", "GatedFusion v3")
    infrastructure_impact = record.infrastructure_impact or rd.get("infrastructure_impact", "N/A")
    emergency_level = record.emergency_level or rd.get("emergency_level", "N/A")
    recommended_actions = record.recommended_actions or rd.get("recommended_actions", [])
    description = _sanitize(record.description or rd.get("description") or rd.get("text", ""))
    event_title = _sanitize(record.event_title or rd.get("event_title", ""))
    disaster_type = _sanitize(disaster_type)
    country = _sanitize(country)
    source = _sanitize(source)
    model_used = _sanitize(model_used)
    infrastructure_impact = _sanitize(infrastructure_impact)
    emergency_level = _sanitize(emergency_level)
    severity = _sanitize(severity)
    priority = _sanitize(priority)
    recommended_actions = [_sanitize(a) for a in recommended_actions]

    # Check for Grad-CAM overlay image
    from backend.config.shared import OUTPUTS_DIR
    explanation_overlay_path = None
    overlay_file = OUTPUTS_DIR / "explanations" / f"{record.id}_overlay.png"
    if overlay_file.exists():
        explanation_overlay_path = str(overlay_file)

    report = {
        # Project header
        "project_name": "CrisisMind",
        "subtitle": "AI Disaster Intelligence Platform",
        "report_title": "AI Assessment Report",
        "version": "1.0.0",
        "generated_at": now.isoformat(),
        "generated_at_formatted": now.strftime("%B %d, %Y at %I:%M %p UTC"),
        # Incident info
        "analysis_id": record.id,
        "analysis_mode": analysis_mode,
        "disaster_type": disaster_type,
        "country": country,
        "source": source,
        "timestamp": record.timestamp.isoformat(),
        "timestamp_formatted": record.timestamp.strftime("%B %d, %Y at %I:%M %p UTC"),
        # Input info
        "description": description,
        "event_title": event_title,
        "filename": record.filename,
        "has_image": False,
        # Prediction
        "severity": severity or "N/A",
        "confidence": confidence,
        "model_used": model_used,
        "processing_time_ms": record.processing_time_ms,
        # Emergency assessment
        "priority": priority or "N/A",
        "infrastructure_impact": infrastructure_impact,
        "emergency_level": emergency_level,
        "recommended_actions": list(recommended_actions),
        # Technical
        "analysis_engine": "AnalysisEngine v3",
        "decision_engine": "DecisionEngine v3",
        "model_version": "3.0.0",
        # Status
        "success": record.success,
        "error": record.error,
        # Explainability
        "explanation_overlay_path": explanation_overlay_path,
    }

    return report
