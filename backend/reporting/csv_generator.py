"""CSV report generation."""

import csv
import io
import logging

logger = logging.getLogger("crisismind.reporting")

FIELDNAMES = [
    "Analysis ID", "Mode", "Severity", "Confidence", "Priority",
    "Emergency Level", "Infrastructure Impact", "Disaster Type",
    "Country", "Source", "Processing Time (ms)", "Timestamp",
    "Event / File", "Description", "Status",
]


def generate_csv(reports: list[dict]) -> str:
    """Generate CSV from a list of report dicts."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()

    for r in reports:
        conf = r.get("confidence")
        conf_str = f"{conf:.2f}" if conf is not None else ""
        writer.writerow({
            "Analysis ID": r.get("analysis_id", ""),
            "Mode": r.get("analysis_mode", ""),
            "Severity": r.get("severity", ""),
            "Confidence": conf_str,
            "Priority": r.get("priority", ""),
            "Emergency Level": r.get("emergency_level", ""),
            "Infrastructure Impact": r.get("infrastructure_impact", ""),
            "Disaster Type": r.get("disaster_type", ""),
            "Country": r.get("country", ""),
            "Source": r.get("source", ""),
            "Processing Time (ms)": f"{r.get('processing_time_ms', 0):.1f}",
            "Timestamp": r.get("timestamp_formatted", ""),
            "Event / File": r.get("event_title") or r.get("filename", ""),
            "Description": r.get("description", ""),
            "Status": "OK" if r.get("success") else f"FAIL: {r.get('error', '')}",
        })

    return output.getvalue()
