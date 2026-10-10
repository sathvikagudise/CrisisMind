"""Cross-cutting validation for normalised DisasterEvent objects.

Removes corrupted records and replaces missing optional values with
sensible defaults so downstream consumers always receive clean data.
"""

import logging
from datetime import datetime

from backend.disaster_intelligence.schemas import DisasterEvent

logger = logging.getLogger("crisismind.validator")


class EventValidator:
    """Validates a DisasterEvent against required-field and quality rules.

    Usage::

        validator = EventValidator()
        if validator.validate(event):
            process(event)
    """

    MIN_TITLE_LENGTH = 3
    MIN_DESCRIPTION_LENGTH = 5
    MAX_COORDINATE_ABS = 180.0

    def validate(self, event: DisasterEvent) -> bool:
        """Return True if the event passes all checks, False otherwise.

        Corrupted records are dropped and logged.
        """
        if not self._check_title(event):
            return False
        if not self._check_description(event):
            return False
        if not self._check_country(event):
            return False
        if not self._check_source_url(event):
            return False
        if not self._check_date(event):
            return False
        if not self._check_coordinates(event):
            return False
        return True

    # ── Individual checks ──────────────────────────────────────────────

    def _check_title(self, event: DisasterEvent) -> bool:
        if not event.title or len(event.title.strip()) < self.MIN_TITLE_LENGTH:
            logger.warning("Validation DROPPED event=%s reason=empty_title", event.event_id)
            return False
        return True

    def _check_description(self, event: DisasterEvent) -> bool:
        if not event.description or len(event.description.strip()) < self.MIN_DESCRIPTION_LENGTH:
            logger.warning("Validation DROPPED event=%s reason=empty_description", event.event_id)
            return False
        return True

    def _check_country(self, event: DisasterEvent) -> bool:
        if not event.country:
            event.country = ""
            logger.debug("Validation DEFAULTS event=%s country=<empty>", event.event_id)
        return True

    def _check_source_url(self, event: DisasterEvent) -> bool:
        if not event.source_url:
            logger.warning("Validation DROPPED event=%s reason=empty_source_url", event.event_id)
            return False
        if not event.source_url.startswith(("http://", "https://")):
            logger.warning("Validation DROPPED event=%s reason=invalid_source_url", event.event_id)
            return False
        return True

    def _check_date(self, event: DisasterEvent) -> bool:
        if event.event_date is not None:
            if not isinstance(event.event_date, datetime):
                logger.warning("Validation DROPPED event=%s reason=invalid_date_type", event.event_id)
                return False
        return True

    def _check_coordinates(self, event: DisasterEvent) -> bool:
        if event.latitude is not None or event.longitude is not None:
            if event.latitude is None or event.longitude is None:
                logger.warning(
                    "Validation PARTIAL event=%s reason=partial_coordinates "
                    "lat=%s lon=%s",
                    event.event_id, event.latitude, event.longitude,
                )
                event.latitude = None
                event.longitude = None
                return True
            try:
                lat = float(event.latitude)
                lon = float(event.longitude)
            except (ValueError, TypeError):
                logger.warning(
                    "Validation FIXED event=%s reason=non_numeric_coordinates "
                    "lat=%s lon=%s",
                    event.event_id, event.latitude, event.longitude,
                )
                event.latitude = None
                event.longitude = None
                return True
            if abs(lat) > 90 or abs(lon) > self.MAX_COORDINATE_ABS:
                logger.warning(
                    "Validation FIXED event=%s reason=out_of_bounds_coordinates "
                    "lat=%s lon=%s",
                    event.event_id, lat, lon,
                )
                event.latitude = None
                event.longitude = None
        return True
