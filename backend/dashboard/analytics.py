from collections import Counter
from backend.dashboard.store import AnalysisRecord


def compute_analytics(records: list[AnalysisRecord]) -> dict:
    severity_dist: dict[str, int] = Counter(
        r.severity for r in records if r.severity and r.success
    )
    priority_dist: dict[str, int] = Counter(
        r.priority for r in records if r.priority and r.success
    )
    disaster_type_dist: dict[str, int] = Counter(
        r.disaster_type for r in records if r.disaster_type and r.success
    )
    analysis_mode_dist: dict[str, int] = Counter(
        r.analysis_mode for r in records if r.analysis_mode and r.success
    )

    confidence_buckets = {"0-25": 0, "26-50": 0, "51-75": 0, "76-100": 0}
    for r in records:
        if r.confidence is not None and r.success:
            c = r.confidence
            if c <= 25:
                confidence_buckets["0-25"] += 1
            elif c <= 50:
                confidence_buckets["26-50"] += 1
            elif c <= 75:
                confidence_buckets["51-75"] += 1
            else:
                confidence_buckets["76-100"] += 1

    time_buckets = {"0-100ms": 0, "101-500ms": 0, "501-2000ms": 0, "2000+ms": 0}
    for r in records:
        if r.success:
            t = r.processing_time_ms
            if t <= 100:
                time_buckets["0-100ms"] += 1
            elif t <= 500:
                time_buckets["101-500ms"] += 1
            elif t <= 2000:
                time_buckets["501-2000ms"] += 1
            else:
                time_buckets["2000+ms"] += 1

    return {
        "severity_distribution": dict(severity_dist),
        "priority_distribution": dict(priority_dist),
        "disaster_type_distribution": dict(disaster_type_dist),
        "analysis_mode_distribution": dict(analysis_mode_dist),
        "confidence_distribution": confidence_buckets,
        "processing_time_distribution": time_buckets,
    }
