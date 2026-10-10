"""Configuration for the Disaster Intelligence module.

All settings are loaded from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class DisasterIntelligenceConfig(BaseSettings):
    """Central configuration for the disaster intelligence module."""

    # ── Provider settings ──────────────────────────────────────────────
    default_provider: str = "nasa"
    max_events_per_request: int = 50
    request_timeout_seconds: int = 30
    retry_count: int = 3
    retry_backoff_seconds: int = 2

    # ── Cache settings ──────────────────────────────────────────────────
    cache_provider: str = "memory"
    cache_duration_seconds: int = 1800
    cache_max_size: int = 1000

    # ── Provider API keys (empty until integration) ─────────────────────
    reliefweb_appname: Optional[str] = None
    reliefweb_api_key: Optional[str] = None
    nasa_api_key: Optional[str] = None
    gdacs_api_key: Optional[str] = None

    # ── Validation ──────────────────────────────────────────────────────
    validate_coordinates: bool = True
    validate_dates: bool = True
    strict_mode: bool = False

    class Config:
        env_prefix = "CM_DI_"


config = DisasterIntelligenceConfig()
