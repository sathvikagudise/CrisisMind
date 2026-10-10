"""Cache interface and built-in MemoryCache implementation."""

from abc import ABC, abstractmethod
from typing import Optional, Any
import time
import logging

logger = logging.getLogger("crisismind.cache")


class Cache(ABC):
    """Cache interface – every backend must implement these four methods."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key.  Returns None when missing."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store a value with an optional TTL (seconds)."""

    @abstractmethod
    def invalidate(self, key: str) -> None:
        """Remove a single key from the cache."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every entry from the cache."""


class MemoryCache(Cache):
    """Thread-safe in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 1800, max_size: int = 1000):
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._store: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        if key not in self._store:
            logger.debug("Cache MISS key=%s", key)
            return None
        if key in self._expiry and now > self._expiry[key]:
            self.invalidate(key)
            logger.debug("Cache EXPIRED key=%s", key)
            return None
        logger.debug("Cache HIT key=%s", key)
        return self._store[key]

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if self._max_size and len(self._store) >= self._max_size:
            self._evict_one()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        self._store[key] = value
        self._expiry[key] = time.time() + ttl
        logger.debug("Cache SET key=%s ttl=%ds", key, ttl)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
        self._expiry.clear()
        logger.debug("Cache CLEARED")

    def _evict_one(self) -> None:
        if not self._store:
            return
        oldest = min(self._expiry, key=lambda k: self._expiry[k])
        self.invalidate(oldest)
