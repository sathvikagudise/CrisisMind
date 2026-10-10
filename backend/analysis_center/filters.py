"""Filter predicates for the Analysis Center."""

from datetime import datetime


def apply_filters(records: list[dict], params: dict) -> list[dict]:
    """Apply multiple filters to a list of normalised records.

    Supported filter keys in *params*:
        - ``analysis_mode`` — exact match on analysis_mode
        - ``severity`` — exact match on severity
        - ``priority`` — exact match on priority
        - ``record_type`` — "manual" / "live" / "batch"
        - ``disaster_type`` — substring match (case-insensitive)
        - ``source`` — substring match (case-insensitive)
        - ``date_from`` — ISO date string (inclusive lower bound)
        - ``date_to`` — ISO date string (inclusive upper bound)
    """
    filtered = records

    if params.get("record_type"):
        filtered = _filter_eq(filtered, "record_type", params["record_type"])

    if params.get("analysis_mode"):
        filtered = _filter_eq(filtered, "analysis_mode", params["analysis_mode"])

    if params.get("severity"):
        filtered = _filter_eq(filtered, "severity", params["severity"])

    if params.get("priority"):
        filtered = _filter_eq(filtered, "priority", params["priority"])

    if params.get("disaster_type"):
        filtered = _filter_substr(filtered, "disaster_type", params["disaster_type"])

    if params.get("source"):
        filtered = _filter_substr(filtered, "source", params["source"])

    if params.get("date_from"):
        date_from = datetime.fromisoformat(params["date_from"])
        filtered = [r for r in filtered if _ts(r) >= date_from]

    if params.get("date_to"):
        date_to = datetime.fromisoformat(params["date_to"])
        filtered = [r for r in filtered if _ts(r) <= date_to]

    return filtered


def _filter_eq(records, field, value):
    return [r for r in records if str(r.get(field, "")).lower() == str(value).lower()]


def _filter_substr(records, field, value):
    val = str(value).lower()
    return [r for r in records if val in str(r.get(field, "")).lower()]


def _ts(record):
    raw = record.get("timestamp")
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            pass
    return datetime.min
