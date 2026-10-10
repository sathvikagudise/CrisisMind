"""Central ReportService — uses the shared ReportBuilder + generators."""

import logging

from backend.dashboard.store import DashboardStore, AnalysisRecord
from backend.reporting.report_builder import build_report_data, _sanitize
from backend.reporting.pdf_generator import generate_pdf
from backend.reporting.csv_generator import generate_csv

logger = logging.getLogger("crisismind.reporting")


class ReportService:
    def __init__(self, dashboard_store: DashboardStore):
        self._store = dashboard_store

    def generate_manual_report(self, analysis_id: str) -> bytes:
        record = self._store.get_by_id(analysis_id)
        if record is None:
            raise ValueError(f"Manual analysis '{analysis_id}' not found")
        report = build_report_data(record)
        return generate_pdf(report)

    def generate_live_report(self, analysis_id: str) -> bytes:
        record = self._store.get_by_id(analysis_id)
        if record is None:
            raise ValueError(f"Live analysis '{analysis_id}' not found")
        report = build_report_data(record)
        return generate_pdf(report)

    def generate_batch_report(self, batch_id: str) -> bytes:
        records = self._store.get_by_batch(batch_id)
        if not records:
            raise ValueError(f"Batch '{batch_id}' not found")
        reports = [build_report_data(r) for r in records]
        return generate_pdf({
            "project_name": "CrisisMind",
            "subtitle": "AI Disaster Intelligence Platform",
            "report_title": _sanitize(f"Batch Analysis Report - {batch_id}"),
            "version": "1.0.0",
            "generated_at": "",
            "generated_at_formatted": "",
            "analysis_id": f"batch_{batch_id}",
            "analysis_mode": "BATCH",
            "disaster_type": "N/A",
            "country": "N/A",
            "source": "batch_processing",
            "timestamp": "",
            "timestamp_formatted": "",
            "description": "",
            "event_title": "",
            "filename": "",
            "has_image": False,
            "severity": "N/A",
            "confidence": None,
            "model_used": "AnalysisEngine v3",
            "processing_time_ms": 0,
            "priority": "N/A",
            "infrastructure_impact": "N/A",
            "emergency_level": "N/A",
            "recommended_actions": [],
            "analysis_engine": "AnalysisEngine v3",
            "decision_engine": "DecisionEngine v3",
            "model_version": "3.0.0",
            "success": True,
            "error": None,
            "_batch_reports": reports,
        })

    def generate_batch_csv(self, batch_id: str) -> str:
        records = self._store.get_by_batch(batch_id)
        if not records:
            raise ValueError(f"Batch '{batch_id}' not found")
        reports = [build_report_data(r) for r in records]
        return generate_csv(reports)
