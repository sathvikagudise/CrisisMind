"""Domain models for the Disaster Intelligence module.

Extends the raw schema layer with provider metadata and lifecycle state.
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

from backend.disaster_intelligence.schemas import DisasterEvent


class ProviderHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ProviderStatus(BaseModel):
    """Runtime status of a single provider."""

    name: str
    health: ProviderHealth = ProviderHealth.HEALTHY
    last_fetch: Optional[datetime] = None
    events_cached: int = 0
    error_message: Optional[str] = None
