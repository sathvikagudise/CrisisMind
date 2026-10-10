"""ReliefWeb API provider — fetches disaster events from reliefweb.int.

NOTE: Since November 2025 the ReliefWeb API v2 requires a pre-approved
appname.  Register at https://apidoc.reliefweb.int/parameters#appname
and set the ``CM_DI_RELIEFWEB_APPNAME`` environment variable.
"""

import logging
from typing import Optional
from datetime import datetime

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

logger = logging.getLogger("crisismind.reliefweb")

BASE_URL = "https://api.reliefweb.int/v2/disasters"
DEFAULT_LIMIT = 20


class ReliefWebProvider(BaseProvider):
    """Fetches disaster events from the ReliefWeb API v2.

    Requires a pre-approved appname set via ``CM_DI_RELIEFWEB_APPNAME``.
    """

    def __init__(self):
        self._appname = config.reliefweb_appname or "crisismind"
        self._client = httpx.Client(
            timeout=config.request_timeout_seconds,
            follow_redirects=True,
        )

    @property
    def provider_name(self) -> str:
        return "reliefweb"

    # ── Public API ─────────────────────────────────────────────────────

    def fetch_events(
        self,
        disaster_type: Optional[str] = None,
        country: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        params = self._build_params(limit)
        filters = self._build_filters(disaster_type, country)

        url = BASE_URL
        if filters:
            params["filter"] = self._make_filter_json(filters)

        logger.info("ReliefWeb GET %s params=%s", url, params)
        data = self._request(url, params)
        items = data.get("data", [])

        if not items:
            raise NoDataAvailable(
                f"ReliefWeb returned 0 events "
                f"(type={disaster_type}, country={country})"
            )

        logger.info("ReliefWeb returned %d items", len(items))
        return items

    def fetch_event(self, event_id: str) -> dict:
        # event_id is stored as "reliefweb-<id>", strip prefix
        rid = event_id.replace("reliefweb-", "", 1)
        url = f"{BASE_URL}/{rid}"
        params = self._build_params(limit=1)
        logger.info("ReliefWeb GET %s", url)
        data = self._request(url, params)
        items = data.get("data", [])
        if not items:
            raise NoDataAvailable(f"ReliefWeb event '{event_id}' not found")
        return items[0]

    def normalize(self, raw_data: dict) -> DisasterEvent:
        fields = raw_data.get("fields", {})
        item_id = str(raw_data.get("id", ""))

        return DisasterEvent(
            event_id=f"reliefweb-{item_id}",
            title=self._safe_str(fields, "name"),
            description=self._safe_html_to_text(fields, "description"),
            disaster_type=self._get_type(fields),
            country=self._get_country(fields),
            latitude=self._safe_float(fields, "latitude"),
            longitude=self._safe_float(fields, "longitude"),
            event_date=self._get_date(fields),
            image_url=self._get_image_url(fields),
            source=self.provider_name,
            source_url=self._safe_str(fields, "url"),
            severity=None,
            metadata={
                "reliefweb_id": item_id,
                "status": fields.get("status"),
                "primary_country": (
                    fields.get("primary_country", {}).get("name")
                    if isinstance(fields.get("primary_country"), dict)
                    else None
                ),
                "glide": fields.get("glide"),
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
                    "ReliefWeb response status=%d attempt=%d/%d",
                    resp.status_code, attempt + 1, config.retry_count,
                )

                if resp.status_code == 429:
                    raise RateLimitExceeded("ReliefWeb rate limit hit (429)")

                if resp.status_code == 403:
                    raise ProviderUnavailable(
                        "ReliefWeb API v2 requires a registered appname. "
                        "Set CM_DI_RELIEFWEB_APPNAME or register at "
                        "https://apidoc.reliefweb.int/parameters#appname"
                    )

                resp.raise_for_status()
                return resp.json()

            except httpx.TimeoutException as exc:
                logger.warning(
                    "ReliefWeb timeout attempt=%d/%d",
                    attempt + 1, config.retry_count,
                )
                if attempt == config.retry_count - 1:
                    raise ProviderUnavailable(
                        f"ReliefWeb timeout after {config.retry_count} retries"
                    ) from exc

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    raise RateLimitExceeded("ReliefWeb rate limit hit (429)") from exc
                if attempt == config.retry_count - 1:
                    raise ProviderUnavailable(
                        f"ReliefWeb HTTP {exc.response.status_code}"
                    ) from exc

            except (httpx.RequestError, ValueError) as exc:
                if attempt == config.retry_count - 1:
                    raise ProviderUnavailable(
                        f"ReliefWeb request failed: {exc}"
                    ) from exc

        raise ProviderUnavailable("ReliefWeb exhausted retries")

    # ── Param builders ─────────────────────────────────────────────────

    def _build_params(self, limit: Optional[int] = None) -> dict:
        return {
            "appname": self._appname,
            "preset": "latest",
            "slim": 1,
            "limit": min(limit or DEFAULT_LIMIT, 100),
        }

    @staticmethod
    def _build_filters(
        disaster_type: Optional[str] = None,
        country: Optional[str] = None,
    ) -> list:
        filters = []
        if disaster_type:
            filters.append({
                "field": "primary_type.name",
                "value": disaster_type.capitalize(),
            })
        if country:
            filters.append({
                "field": "country.name",
                "value": country,
            })
        return filters

    @staticmethod
    def _make_filter_json(filters: list) -> dict:
        return {
            "operator": "AND",
            "conditions": [
                {"field": f["field"], "value": f["value"]} for f in filters
            ],
        }

    # ── Field mappers ──────────────────────────────────────────────────

    @staticmethod
    def _safe_str(fields: dict, key: str, default: str = "") -> str:
        val = fields.get(key)
        return str(val) if val is not None else default

    @staticmethod
    def _safe_float(fields: dict, key: str) -> Optional[float]:
        val = fields.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_html_to_text(fields: dict, key: str) -> str:
        import re
        import html as html_mod
        val = fields.get(key, "")
        if not isinstance(val, str):
            return ""
        clean = re.sub(r"<[^>]+>", " ", val)
        clean = html_mod.unescape(clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @staticmethod
    def _get_type(fields: dict) -> str:
        primary = fields.get("primary_type")
        if isinstance(primary, dict):
            return primary.get("name", "")
        if isinstance(primary, str):
            return primary
        return ""

    @staticmethod
    def _get_country(fields: dict) -> str:
        countries = fields.get("country", [])
        if isinstance(countries, list) and countries:
            first = countries[0]
            if isinstance(first, dict):
                return first.get("name", "")
            return str(first)
        return ""

    @staticmethod
    def _get_date(fields: dict) -> Optional[datetime]:
        date_obj = fields.get("date")
        if isinstance(date_obj, dict):
            created = date_obj.get("created")
            if created:
                try:
                    return datetime.fromisoformat(created.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def _get_image_url(fields: dict) -> Optional[str]:
        image = fields.get("image")
        if isinstance(image, dict):
            url = image.get("url") or image.get("original")
            if url:
                return url
        map_url = fields.get("map_url")
        if map_url:
            return map_url
        return None
