import logging
from datetime import datetime
from typing import Optional, Literal

logger = logging.getLogger("crisismind.dashboard")


class AnalysisRecord:
    def __init__(
        self,
        record_id: str,
        record_type: Literal["manual", "live", "batch"],
        timestamp: Optional[datetime] = None,
        source: str = "",
        severity: Optional[str] = None,
        confidence: Optional[float] = None,
        priority: Optional[str] = None,
        analysis_mode: Optional[str] = None,
        disaster_type: Optional[str] = None,
        processing_time_ms: float = 0,
        filename: Optional[str] = None,
        event_title: Optional[str] = None,
        batch_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        raw_data: Optional[dict] = None,
        description: Optional[str] = None,
        country: Optional[str] = None,
        model_used: Optional[str] = None,
        infrastructure_impact: Optional[str] = None,
        emergency_level: Optional[str] = None,
        recommended_actions: Optional[list[str]] = None,
    ):
        self.id = record_id
        self.type = record_type
        self.timestamp = timestamp or datetime.utcnow()
        self.source = source
        self.severity = severity
        self.confidence = confidence
        self.priority = priority
        self.analysis_mode = analysis_mode
        self.disaster_type = disaster_type
        self.processing_time_ms = processing_time_ms
        self.filename = filename
        self.event_title = event_title
        self.batch_id = batch_id
        self.success = success
        self.error = error
        self.raw_data = raw_data
        self.description = description
        self.country = country
        self.model_used = model_used
        self.infrastructure_impact = infrastructure_impact
        self.emergency_level = emergency_level
        self.recommended_actions = recommended_actions or []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "severity": self.severity,
            "confidence": self.confidence,
            "priority": self.priority,
            "analysis_mode": self.analysis_mode,
            "disaster_type": self.disaster_type,
            "processing_time_ms": self.processing_time_ms,
            "filename": self.filename,
            "event_title": self.event_title,
            "batch_id": self.batch_id,
            "success": self.success,
            "error": self.error,
            "description": self.description,
            "country": self.country,
            "model_used": self.model_used,
            "infrastructure_impact": self.infrastructure_impact,
            "emergency_level": self.emergency_level,
            "recommended_actions": list(self.recommended_actions),
        }


class DashboardStore:
    def __init__(self):
        self._records: list[AnalysisRecord] = []

    def track(self, record: AnalysisRecord) -> None:
        self._records.append(record)
        logger.debug("Dashboard tracked: type=%s id=%s severity=%s", record.type, record.id, record.severity)

    def get_all(self) -> list[AnalysisRecord]:
        return list(self._records)

    def get_by_type(self, record_type: str) -> list[AnalysisRecord]:
        return [r for r in self._records if r.type == record_type]

    def get_by_id(self, record_id: str) -> Optional[AnalysisRecord]:
        for r in self._records:
            if r.id == record_id:
                return r
        return None

    def get_by_batch(self, batch_id: str) -> list[AnalysisRecord]:
        return [r for r in self._records if r.batch_id == batch_id]

    def clear(self) -> None:
        self._records.clear()
