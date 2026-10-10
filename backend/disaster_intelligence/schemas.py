"""Unified DisasterEvent schema.

Every provider normalises its response into this single format so that
downstream consumers (AI pipeline, frontend, etc.) never depend on the
source API.
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class DisasterEvent(BaseModel):
    """Standardised disaster event — the lingua franca of this module."""

    event_id: str
    title: str
    description: str
    disaster_type: str
    country: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    event_date: Optional[datetime] = None
    image_url: Optional[str] = None
    source: str
    source_url: str
    severity: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DisasterEventList(BaseModel):
    """Wrapper for a list of events returned by the service layer."""

    events: list[DisasterEvent]
    total: int
    source: str
