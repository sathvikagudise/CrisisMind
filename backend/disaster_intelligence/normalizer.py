"""Normalisation layer — converts arbitrary provider responses into DisasterEvent.

The AI pipeline never knows which provider generated the data.
"""

import html as html_mod
from typing import Any
from datetime import datetime

from backend.disaster_intelligence.schemas import DisasterEvent
from backend.disaster_intelligence.exceptions import InvalidResponse


class EventNormalizer:
    """Translates raw provider dicts into standardised DisasterEvent objects.

    Each provider calls ``normalize()`` with its own raw payload; this
    class delegates field extraction to the provider via its ``normalize``
    method and performs cross-cutting validation.
    """

    def normalize(self, raw: dict, provider_name: str) -> DisasterEvent:
        """High-level entry point for normalisation.

        Args:
            raw: Raw dictionary returned by a provider's API call.
            provider_name: Source identifier (e.g. 'ReliefWeb').

        Returns:
            A validated DisasterEvent.

        Raises:
            InvalidResponse: The raw data is missing required fields or
                cannot be coerced into a DisasterEvent.
        """
        try:
            event = DisasterEvent(
                event_id=self._safe_str(raw, "event_id"),
                title=html_mod.unescape(self._safe_str(raw, "title")),
                description=html_mod.unescape(self._safe_str(raw, "description")),
                disaster_type=self._safe_str(raw, "disaster_type"),
                country=html_mod.unescape(self._safe_str(raw, "country")),
                latitude=self._safe_float(raw, "latitude"),
                longitude=self._safe_float(raw, "longitude"),
                event_date=self._safe_datetime(raw, "event_date"),
                image_url=self._safe_str(raw, "image_url"),
                source=provider_name,
                source_url=self._safe_str(raw, "source_url"),
                severity=self._safe_float(raw, "severity"),
                metadata=raw.get("metadata", {}),
            )
            return event
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidResponse(
                f"Cannot normalise response from '{provider_name}': {exc}"
            ) from exc

    # ── Helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _safe_str(data: dict, key: str, default: str = "") -> str:
        val = data.get(key)
        return str(val) if val is not None else default

    @staticmethod
    def _safe_float(data: dict, key: str) -> float | None:
        val = data.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_datetime(data: dict, key: str) -> datetime | None:
        val = data.get(key)
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        return None
