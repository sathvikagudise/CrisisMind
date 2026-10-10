"""AnalysisCenterService — aggregates manual, live, and batch analyses."""

import logging
from datetime import datetime

from backend.dashboard.store import DashboardStore
from backend.disaster_intelligence.store import AnalysisStore
from backend.analysis_center.filters import apply_filters
from backend.analysis_center.search import search_records
from backend.analysis_center.pagination import paginate

logger = logging.getLogger("crisismind.analysis_center")


def _normalize_dashboard_record(record) -> dict:
    """Convert an AnalysisRecord to a unified dict."""
    d = record.to_dict()
    return {
        "analysis_id": d["id"],
        "record_type": d["type"],
        "timestamp": d["timestamp"],
        "source": d["source"],
        "severity": d["severity"],
        "confidence": d["confidence"],
        "priority": d["priority"],
        "analysis_mode": d["analysis_mode"],
        "disaster_type": d["disaster_type"],
        "processing_time_ms": d["processing_time_ms"],
        "filename": d["filename"],
        "event_title": d["event_title"],
        "batch_id": d["batch_id"],
        "description": d["description"],
        "country": d["country"],
        "model_used": d["model_used"],
        "infrastructure_impact": d["infrastructure_impact"],
        "emergency_level": d["emergency_level"],
        "recommended_actions": d["recommended_actions"],
        "success": d["success"],
        "error": d["error"],
        "has_explanation": False,
    }


def _normalize_live_record(event_id: str, result: dict, timestamp: datetime) -> dict:
    """Convert an AnalysisStore live-analysis entry to a unified dict."""
    event = result.get("event", {})
    analysis = result.get("analysis", {})
    return {
        "analysis_id": event_id,
        "record_type": "live",
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
        "source": "disaster_intelligence",
        "severity": analysis.get("severity"),
        "confidence": analysis.get("confidence"),
        "priority": analysis.get("priority"),
        "analysis_mode": result.get("analysis_mode", "live"),
        "disaster_type": event.get("type"),
        "processing_time_ms": result.get("processing_time_ms", 0),
        "filename": None,
        "event_title": event.get("title"),
        "batch_id": None,
        "description": event.get("description"),
        "country": event.get("country"),
        "model_used": result.get("model_used"),
        "infrastructure_impact": analysis.get("infrastructure_impact"),
        "emergency_level": analysis.get("emergency_level"),
        "recommended_actions": analysis.get("recommended_actions", []),
        "success": result.get("success", True),
        "error": result.get("error"),
        "has_explanation": False,
    }


SORT_FUNCTIONS = {
    "newest": lambda r: _sort_ts(r, reverse=True),
    "oldest": lambda r: _sort_ts(r, reverse=False),
    "highest_confidence": lambda r: (r.get("confidence") or 0) * -1,
    "lowest_confidence": lambda r: (r.get("confidence") or 0),
    "highest_priority": lambda r: _priority_sort_key(r, reverse=True),
}


def _sort_ts(record, reverse=False):
    raw = record.get("timestamp")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return datetime.min
    if isinstance(raw, datetime):
        return raw
    return datetime.min


def _priority_sort_key(record, reverse=False):
    order = {"high": 3, "medium": 2, "low": 1, "critical": 4}
    pri = str(record.get("priority", "")).lower()
    score = order.get(pri, 0)
    return score * -1 if reverse else score


class AnalysisCenterService:
    """Unified service for browsing all completed analyses."""

    def __init__(self, dashboard_store: DashboardStore,
                 analysis_store: AnalysisStore):
        self._ds = dashboard_store
        self._as = analysis_store

    def _all_records(self) -> list[dict]:
        """Return all normalised records from both stores, newest first."""
        records = []

        # DashboardStore records (manual + batch)
        for r in self._ds.get_all():
            nr = _normalize_dashboard_record(r)
            # Check if explanation exists
            from pathlib import Path
            from backend.config.shared import OUTPUTS_DIR
            overlay = OUTPUTS_DIR / "explanations" / f"{nr['analysis_id']}_overlay.png"
            nr["has_explanation"] = overlay.exists()
            records.append(nr)

        # AnalysisStore records (live disaster analyses)
        for event_id in self._as.get_analyzed_ids():
            result = self._as.get(event_id)
            ts = self._as.get_entry_timestamp(event_id) or datetime.utcnow()
            if result:
                records.append(_normalize_live_record(event_id, result, ts))

        # Sort newest first by default
        records.sort(key=lambda r: _sort_ts(r), reverse=True)
        return records

    def list_analyses(self, page: int = 1, per_page: int = 20,
                      sort_by: str = "newest",
                      **filters) -> dict:
        """Paginated, filtered, sorted list of all analyses."""
        records = self._all_records()

        # Apply search
        query = filters.pop("query", None)
        if query:
            records = search_records(records, query)

        # Apply filters
        records = apply_filters(records, filters)

        # Apply sort
        sort_fn = SORT_FUNCTIONS.get(sort_by, SORT_FUNCTIONS["newest"])
        records.sort(key=sort_fn)

        return paginate(records, page=page, per_page=per_page)

    def get_analysis(self, analysis_id: str) -> dict | None:
        """Return a single analysis by ID."""
        for r in self._all_records():
            if r["analysis_id"] == analysis_id:
                return r
        return None

    def search(self, query: str, page: int = 1, per_page: int = 20,
               sort_by: str = "newest") -> dict:
        """Full-text search across all analyses."""
        records = self._all_records()
        records = search_records(records, query)
        sort_fn = SORT_FUNCTIONS.get(sort_by, SORT_FUNCTIONS["newest"])
        records.sort(key=sort_fn)
        return paginate(records, page=page, per_page=per_page)

    def export_csv(self, **filters) -> str:
        """Generate CSV string of filtered results."""
        import csv, io

        records = self._all_records()
        query = filters.pop("query", None)
        if query:
            records = search_records(records, query)
        records = apply_filters(records, filters)

        fields = [
            "analysis_id", "record_type", "timestamp", "source",
            "severity", "confidence", "priority", "analysis_mode",
            "disaster_type", "country", "description", "event_title",
            "model_used", "processing_time_ms",
        ]

        buff = io.StringIO()
        writer = csv.DictWriter(buff, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            if isinstance(row.get("confidence"), (int, float)):
                row["confidence"] = f"{row['confidence']:.2f}"
            writer.writerow(row)
        return buff.getvalue()

    def get_statistics(self) -> dict:
        """Aggregate statistics across all analyses."""
        records = self._all_records()
        total = len(records)
        by_type = {}
        by_severity = {}
        by_priority = {}

        for r in records:
            rt = r.get("record_type", "unknown")
            by_type[rt] = by_type.get(rt, 0) + 1

            sev = r.get("severity", "Unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1

            pri = r.get("priority", "Unknown")
            by_priority[pri] = by_priority.get(pri, 0) + 1

        confs = [r.get("confidence") or 0 for r in records if r.get("confidence") is not None]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        times = [r.get("processing_time_ms") or 0 for r in records]
        avg_time = sum(times) / len(times) if times else 0.0

        return {
            "total_analyses": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "by_priority": by_priority,
            "average_confidence": round(avg_conf, 2),
            "average_processing_time_ms": round(avg_time, 2),
        }
