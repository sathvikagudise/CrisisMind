from typing import Optional
from backend.dashboard.store import AnalysisRecord


def compute_statistics(records: list[AnalysisRecord]) -> dict:
    total = len(records)
    manual = sum(1 for r in records if r.type == "manual")
    live = sum(1 for r in records if r.type == "live")
    batch = sum(1 for r in records if r.type == "batch")

    high = sum(1 for r in records if r.priority == "High")
    medium = sum(1 for r in records if r.priority == "Medium")
    low = sum(1 for r in records if r.priority == "Low")

    confidences = [r.confidence for r in records if r.confidence is not None and r.success]
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    times = [r.processing_time_ms for r in records if r.success]
    avg_time = round(sum(times) / len(times), 2) if times else 0.0

    disasters = [r.disaster_type for r in records if r.disaster_type]
    most_common_type = max(set(disasters), key=disasters.count) if disasters else "N/A"

    severities = [r.severity for r in records if r.severity]
    most_common_severity = max(set(severities), key=severities.count) if severities else "N/A"

    return {
        "total_analyses": total,
        "manual": manual,
        "live": live,
        "batch": batch,
        "high_priority": high,
        "medium_priority": medium,
        "low_priority": low,
        "average_confidence": avg_confidence,
        "average_processing_time_ms": avg_time,
        "most_common_disaster_type": most_common_type,
        "most_common_severity": most_common_severity,
    }
