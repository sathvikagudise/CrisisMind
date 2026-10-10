"""Service layer — orchestrates providers, normalisation, validation, and caching.

The service is the main entry point for the rest of the application.
It loads a provider, fetches raw events, normalises them, validates
them, caches the result, and returns standardised ``DisasterEvent``
objects — no AI or frontend code is touched.
"""

import logging
from typing import Optional

from backend.disaster_intelligence.config import config
from backend.disaster_intelligence.schemas import DisasterEvent, DisasterEventList
from backend.disaster_intelligence.cache import Cache, MemoryCache
from backend.disaster_intelligence.normalizer import EventNormalizer
from backend.disaster_intelligence.validators import EventValidator
from backend.disaster_intelligence.exceptions import (
    InvalidConfiguration,
    NoDataAvailable,
    ProviderUnavailable,
    RateLimitExceeded,
)
from backend.disaster_intelligence.providers.base_provider import BaseProvider

logger = logging.getLogger("crisismind.disaster_service")


class DisasterIntelligenceService:
    """Orchestrates the full fetch → normalise → validate → cache flow.

    Usage::

        service = DisasterIntelligenceService()
        result = service.get_events(disaster_type="flood", country="Philippines")
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        normalizer: Optional[EventNormalizer] = None,
        validator: Optional[EventValidator] = None,
    ):
        self._cache = cache or MemoryCache(
            default_ttl=config.cache_duration_seconds,
            max_size=config.cache_max_size,
        )
        self._normalizer = normalizer or EventNormalizer()
        self._validator = validator or EventValidator()
        self._provider: Optional[BaseProvider] = None

    # ── Provider loading ────────────────────────────────────────────────
    def load_provider(self, name: Optional[str] = None) -> None:
        """Load a provider by name (or fall back to the default)."""
        provider_name = name or config.default_provider

        if provider_name == "reliefweb":
            from backend.disaster_intelligence.providers.reliefweb_provider import (
                ReliefWebProvider,
            )
            self._provider = ReliefWebProvider()
        elif provider_name == "nasa":
            from backend.disaster_intelligence.providers.nasa_provider import (
                NASAProvider,
            )
            self._provider = NASAProvider()
        elif provider_name == "gdacs":
            from backend.disaster_intelligence.providers.gdacs_provider import (
                GDACSProvider,
            )
            self._provider = GDACSProvider()
        else:
            raise InvalidConfiguration(f"Unknown provider: '{provider_name}'")

    @property
    def provider(self) -> BaseProvider:
        if self._provider is None:
            self.load_provider()
        return self._provider

    # ── Core operations ─────────────────────────────────────────────────
    def get_events(
        self,
        disaster_type: Optional[str] = None,
        country: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> DisasterEventList:
        """Fetch, normalise, validate, and return events from the active provider."""
        prov = self.provider
        cache_key = self._build_cache_key(prov.provider_name, disaster_type, country)

        # 1. Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info(
                "Cache HIT provider=%s key=%s events=%d",
                prov.provider_name, cache_key, cached.total,
            )
            return cached

        logger.info(
            "Cache MISS provider=%s key=%s — calling API",
            prov.provider_name, cache_key,
        )

        # 2. Fetch from provider
        try:
            raw_items = prov.fetch_events(
                disaster_type=disaster_type,
                country=country,
                limit=limit or config.max_events_per_request,
            )
        except (ProviderUnavailable, RateLimitExceeded, NoDataAvailable):
            raise

        # 3. Normalise via the provider's own field-mapping
        normalised: list[DisasterEvent] = []
        for item in raw_items:
            try:
                event = prov.normalize(item)
                normalised.append(event)
            except Exception as exc:
                logger.warning("Normalisation failed for item: %s", exc)

        logger.info(
            "Normalised %d/%d items from %s",
            len(normalised), len(raw_items), prov.provider_name,
        )

        # 4. Validate (cross-cutting)
        validated: list[DisasterEvent] = []
        validation_failures = 0
        for event in normalised:
            if self._validator.validate(event):
                validated.append(event)
            else:
                validation_failures += 1

        if validation_failures:
            logger.warning(
                "Validation dropped %d/%d events",
                validation_failures, len(normalised),
            )

        # 5. Provider-specific validation
        final: list[DisasterEvent] = []
        for event in validated:
            if prov.validate(event):
                final.append(event)

        result = DisasterEventList(
            events=final,
            total=len(final),
            source=prov.provider_name,
        )

        logger.info(
            "Fetched %d events from %s (valid=%d dropped=%d)",
            len(raw_items), prov.provider_name, len(final),
            len(raw_items) - len(final),
        )

        # 6. Store in cache
        self._cache.set(cache_key, result)

        return result

    def get_event(self, event_id: str) -> DisasterEvent:
        """Fetch a single event by ID."""
        prov = self.provider

        cache_key = f"{prov.provider_name}:event:{event_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("Cache HIT event=%s", event_id)
            return cached

        logger.info("Cache MISS event=%s — calling API", event_id)
        raw = prov.fetch_event(event_id)
        event = prov.normalize(raw)
        self._validator.validate(event)
        prov.validate(event)

        self._cache.set(cache_key, event)
        return event

    # ── Internals ───────────────────────────────────────────────────────
    @staticmethod
    def _build_cache_key(
        provider: str,
        disaster_type: Optional[str],
        country: Optional[str],
    ) -> str:
        parts = [provider, "events"]
        if disaster_type:
            parts.append(disaster_type)
        if country:
            parts.append(country)
        return ":".join(parts)
