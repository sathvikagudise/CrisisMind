from backend.dashboard.store import AnalysisRecord


def build_timeline(records: list[AnalysisRecord], limit: int = 20) -> list[dict]:
    sorted_records = sorted(records, key=lambda r: r.timestamp, reverse=True)
    result = []
    for r in sorted_records[:limit]:
        entry = {
            "id": r.id,
            "type": r.type,
            "timestamp": r.timestamp.isoformat(),
            "severity": r.severity,
            "confidence": r.confidence,
            "priority": r.priority,
            "analysis_mode": r.analysis_mode,
            "disaster_type": r.disaster_type,
            "processing_time_ms": r.processing_time_ms,
            "filename": r.filename,
            "event_title": r.event_title,
            "batch_id": r.batch_id,
            "success": r.success,
            "error": r.error,
        }
        result.append(entry)
    return result
