"""Abstract base provider — every data source must inherit from this."""

from abc import ABC, abstractmethod
from typing import Optional

from backend.disaster_intelligence.schemas import DisasterEvent
from backend.disaster_intelligence.exceptions import (
    ProviderUnavailable,
    InvalidResponse,
    RateLimitExceeded,
    NoDataAvailable,
)


class BaseProvider(ABC):
    """Interface that all disaster-data providers must implement.

    Subclasses override fetch_events / fetch_event to call their
    respective external API, return raw dicts, and implement
    ``normalize()`` to convert a single raw dict into a ``DisasterEvent``.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name, e.g. 'ReliefWeb' or 'GDACS'."""

    @abstractmethod
    def fetch_events(
        self,
        disaster_type: Optional[str] = None,
        country: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Return raw API response dicts for recent disaster events.

        Raises:
            ProviderUnavailable: API is unreachable.
            RateLimitExceeded:   API quota exceeded.
            InvalidResponse:     Response cannot be parsed.
            NoDataAvailable:     Zero results for the given filters.
        """

    @abstractmethod
    def fetch_event(self, event_id: str) -> dict:
        """Return a single raw API response dict by event ID.

        Raises:
            ProviderUnavailable: API is unreachable.
            InvalidResponse:     Response cannot be parsed.
            NoDataAvailable:     No event matches *event_id*.
        """

    @abstractmethod
    def normalize(self, raw_data: dict) -> DisasterEvent:
        """Convert a single raw API response dict into a DisasterEvent.

        Every provider implements its own field-mapping logic here so
        that downstream code never depends on the source API shape.
        """

    @abstractmethod
    def validate(self, event: DisasterEvent) -> bool:
        """Provider-specific validation after normalisation.

        Return ``True`` if the event passes all provider-specific checks,
        ``False`` otherwise (the event is silently dropped).
        """
