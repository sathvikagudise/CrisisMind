"""GDACS provider skeleton.

Integrates with the Global Disaster Alert and Coordination System (GDACS).
Full implementation will be added in a future integration phase.
"""

from typing import Optional

from backend.disaster_intelligence.schemas import DisasterEvent
from backend.disaster_intelligence.providers.base_provider import BaseProvider


class GDACSProvider(BaseProvider):
    """Disaster data provider for GDACS."""

    @property
    def provider_name(self) -> str:
        return "gdacs"

    def fetch_events(
        self,
        disaster_type: Optional[str] = None,
        country: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[DisasterEvent]:
        raise NotImplementedError("GDACS integration pending")

    def fetch_event(self, event_id: str) -> DisasterEvent:
        raise NotImplementedError("GDACS integration pending")

    def normalize(self, raw_data: dict) -> DisasterEvent:
        raise NotImplementedError("GDACS integration pending")

    def validate(self, event: DisasterEvent) -> bool:
        return True
