"""Search logic for the Analysis Center."""


def search_records(records: list[dict], query: str) -> list[dict]:
    """Return records whose text fields match *query* (case-insensitive).

    Searched fields: ``analysis_id``, ``country``, ``disaster_type``,
    ``description``, ``source``, ``event_title``.
    """
    if not query or not query.strip():
        return records

    q = query.strip().lower()
    searched_fields = ["analysis_id", "country", "disaster_type",
                       "description", "source", "event_title", "filename"]

    def _matches(r):
        for field in searched_fields:
            val = r.get(field)
            if val and q in str(val).lower():
                return True
        return False

    return [r for r in records if _matches(r)]
