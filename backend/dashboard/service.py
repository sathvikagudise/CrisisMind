import logging
from typing import Optional

from backend.dashboard.store import DashboardStore, AnalysisRecord
from backend.dashboard.statistics import compute_statistics
from backend.dashboard.charts import build_chart_data
from backend.dashboard.timeline import build_timeline
from backend.dashboard.analytics import compute_analytics

logger = logging.getLogger("crisismind.dashboard")


class DashboardService:
    def __init__(self, dashboard_store: DashboardStore):
        self._store = dashboard_store

    def get_summary(self) -> dict:
        records = self._store.get_all()
        stats = compute_statistics(records)
        return {
            "total_analyses": stats["total_analyses"],
            "manual": stats["manual"],
            "live": stats["live"],
            "batch": stats["batch"],
            "high_priority": stats["high_priority"],
            "average_confidence": stats["average_confidence"],
            "average_processing_time_ms": stats["average_processing_time_ms"],
        }

    def get_statistics(self) -> dict:
        records = self._store.get_all()
        return compute_statistics(records)

    def get_charts(self) -> dict:
        records = self._store.get_all()
        return build_chart_data(records)

    def get_timeline(self, limit: int = 20) -> list[dict]:
        records = self._store.get_all()
        return build_timeline(records, limit)

    def get_recent(self, limit: int = 10) -> list[dict]:
        records = self._store.get_all()
        sorted_records = sorted(records, key=lambda r: r.timestamp, reverse=True)
        return [r.to_dict() for r in sorted_records[:limit]]

    def track_manual(self, record: AnalysisRecord) -> None:
        self._store.track(record)

    def track_batch(self, record: AnalysisRecord) -> None:
        self._store.track(record)

    def track_live(self, record: AnalysisRecord) -> None:
        self._store.track(record)
