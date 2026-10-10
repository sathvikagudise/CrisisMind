"""NASA EONET provider — fetches natural events from eoonet.gsfc.nasa.gov.

NASA's Earth Observatory Natural Event Tracker (EONET) provides a free,
open API that requires no authentication.  This provider is used as a
working reference while awaiting ReliefWeb appname approval.
"""

import logging
from typing import Optional
from datetime import datetime
import html as html_mod

import httpx

from backend.disaster_intelligence.config import config
from backend.disaster_intelligence.schemas import DisasterEvent
from backend.disaster_intelligence.providers.base_provider import BaseProvider
from backend.disaster_intelligence.exceptions import (
    ProviderUnavailable,
    InvalidResponse,
    RateLimitExceeded,
    NoDataAvailable,
)

logger = logging.getLogger("crisismind.nasa")

BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
DEFAULT_LIMIT = 20

# Map NASA category IDs to disaster_type strings
CATEGORY_MAP = {
    "drought": "Drought",
    "dustHaze": "Dust/Haze",
    "earthquakes": "Earthquake",
    "floods": "Flood",
    "landslides": "Landslide",
    "manmade": "Man-made",
    "seaLakeIce": "Sea/Lake Ice",
    "severeStorms": "Severe Storm",
    "snow": "Snow",
    "temperatureExtremes": "Temperature Extremes",
    "volcanoes": "Volcano",
    "waterColor": "Water Color",
    "wildfires": "Wildfire",
}


class NASAProvider(BaseProvider):
    """Fetches natural disaster events from NASA EONET (free, no auth)."""

    def __init__(self):
        self._client = httpx.Client(
            timeout=config.request_timeout_seconds,
            follow_redirects=True,
        )

    @property
    def provider_name(self) -> str:
        return "nasa"

    # ── Public API ─────────────────────────────────────────────────────

    def fetch_events(
        self,
        disaster_type: Optional[str] = None,
        country: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        params = {"limit": min(limit or DEFAULT_LIMIT, 100), "status": "open"}
        if disaster_type:
            # Map our type to NASA category ID
            for cat_id, name in CATEGORY_MAP.items():
                if name.lower() == disaster_type.lower():
                    params["category"] = cat_id
                    break

        logger.info("NASA GET %s params=%s", BASE_URL, params)
        data = self._request(BASE_URL, params)
        items = data.get("events", [])

        # NASA returns all events globally; country filter is client-side
        if country:
            filtered = []
            country_lower = country.lower()
            for item in items:
                geo_country = self._extract_country(item)
                if geo_country and country_lower in geo_country.lower():
                    filtered.append(item)
            items = filtered

        if not items:
            raise NoDataAvailable(
                f"NASA returned 0 events (type={disaster_type}, country={country})"
            )

        logger.info("NASA returned %d items", len(items))
        return items

    def fetch_event(self, event_id: str) -> dict:
        # event_id stored as "nasa-<id>", strip prefix
        eid = event_id.replace("nasa-", "", 1)
        url = f"{BASE_URL}/{eid}"
        logger.info("NASA GET %s", url)
        data = self._request(url, {})
        return data

    def normalize(self, raw_data: dict) -> DisasterEvent:
        item_id = str(raw_data.get("id", ""))
        title = raw_data.get("title", "")

        # Extract first non-null geometry entry
        geometry = raw_data.get("geometry", [])
        first_geo = geometry[0] if geometry else {}
        coords = first_geo.get("coordinates", [None, None])
        event_date = self._parse_date(first_geo.get("date"))

        # Categories
        categories = raw_data.get("categories", [])
        disaster_type = (
            categories[0].get("title", "") if categories else ""
        )

        # Source URL
        sources = raw_data.get("sources", [])
        source_url = sources[0].get("url", "") if sources else ""

        # Description
        description = raw_data.get("description") or title or "No description available"

        # Country — not directly in EONET, extract from geometry/title heuristics
        country = self._extract_country(raw_data) or ""

        return DisasterEvent(
            event_id=f"nasa-{item_id}",
            title=html_mod.unescape(title),
            description=html_mod.unescape(description),
            disaster_type=disaster_type,
            country=html_mod.unescape(country),
            latitude=float(coords[1]) if coords and len(coords) > 1 and coords[1] is not None else None,
            longitude=float(coords[0]) if coords and coords[0] is not None else None,
            event_date=event_date,
            image_url=None,
            source=self.provider_name,
            source_url=source_url,
            severity=None,
            metadata={
                "nasa_id": item_id,
                "categories": [c.get("title") for c in categories],
                "geometry_count": len(geometry),
                "closed": raw_data.get("closed"),
            },
        )

    def validate(self, event: DisasterEvent) -> bool:
        return True

    # ── HTTP layer ─────────────────────────────────────────────────────

    def _request(self, url: str, params: dict) -> dict:
        for attempt in range(config.retry_count):
            try:
                resp = self._client.get(url, params=params)
                logger.info(
                    "NASA response status=%d attempt=%d/%d",
                    resp.status_code, attempt + 1, config.retry_count,
                )
                resp.raise_for_status()
                return resp.json()

            except httpx.TimeoutException as exc:
                logger.warning(
                    "NASA timeout attempt=%d/%d", attempt + 1, config.retry_count,
                )
                if attempt == config.retry_count - 1:
                    raise ProviderUnavailable(
                        f"NASA timeout after {config.retry_count} retries"
                    ) from exc

            except httpx.HTTPStatusError as exc:
                if attempt == config.retry_count - 1:
                    raise ProviderUnavailable(
                        f"NASA HTTP {exc.response.status_code}"
                    ) from exc

            except (httpx.RequestError, ValueError) as exc:
                if attempt == config.retry_count - 1:
                    raise ProviderUnavailable(
                        f"NASA request failed: {exc}"
                    ) from exc

        raise ProviderUnavailable("NASA exhausted retries")

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_country(raw: dict) -> Optional[str]:
        """Best-effort country extraction from NASA event data."""
        # Try 'countries' field
        countries = raw.get("countries")
        if isinstance(countries, list) and countries:
            if isinstance(countries[0], dict):
                return countries[0].get("name", "")
            return str(countries[0])

        # Search for known country names in title
        title = raw.get("title", "")
        known_countries = [
            "USA", "United States", "Canada", "Mexico", "Brazil", "Argentina",
            "Chile", "India", "China", "Japan", "Australia", "Indonesia",
            "Philippines", "Russia", "France", "Germany", "Italy", "Spain",
            "UK", "United Kingdom", "Turkey", "Iran", "Iraq", "Afghanistan",
            "Pakistan", "Bangladesh", "Vietnam", "Thailand", "Myanmar",
            "South Africa", "Nigeria", "Kenya", "Ethiopia", "Somalia",
            "New Zealand", "Peru", "Colombia", "Venezuela",
        ]
        title_lower = title.lower()
        for c in known_countries:
            if c.lower() in title_lower:
                return c

        # Fallback: last comma-separated part of title (often a US state)
        parts = [p.strip() for p in title.split(",")]
        if len(parts) >= 2:
            candidate = parts[-1]
            if len(candidate) <= 20 and not candidate.startswith("http"):
                return candidate
        return None
