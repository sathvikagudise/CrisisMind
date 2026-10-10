from backend.dashboard.analytics import compute_analytics
from backend.dashboard.store import AnalysisRecord


def build_chart_data(records: list[AnalysisRecord]) -> dict:
    analytics = compute_analytics(records)

    severity_chart = [
        {"name": k, "value": v}
        for k, v in sorted(analytics["severity_distribution"].items())
    ]
    priority_chart = [
        {"name": k, "value": v}
        for k, v in sorted(analytics["priority_distribution"].items(),
                         key=lambda x: {"High": 0, "Medium": 1, "Low": 2}.get(x[0], 99))
    ]
    disaster_type_chart = [
        {"name": k if len(k) <= 15 else k[:12] + "...", "value": v}
        for k, v in sorted(analytics["disaster_type_distribution"].items(),
                           key=lambda x: x[1], reverse=True)
    ]
    analysis_mode_chart = [
        {"name": k, "value": v}
        for k, v in sorted(analytics["analysis_mode_distribution"].items())
    ]
    confidence_chart = [
        {"name": k, "value": v}
        for k, v in analytics["confidence_distribution"].items()
    ]
    processing_time_chart = [
        {"name": k, "value": v}
        for k, v in analytics["processing_time_distribution"].items()
    ]

    return {
        "severity": severity_chart,
        "priority": priority_chart,
        "disaster_types": disaster_type_chart,
        "analysis_modes": analysis_mode_chart,
        "confidence": confidence_chart,
        "processing_time": processing_time_chart,
    }
