"""FastAPI routes for the Disaster Intelligence module.

Provides read-only endpoints to list and retrieve disaster events
from the configured provider, plus an analysis endpoint that bridges
events into the Unified Analysis Engine.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from backend.dashboard import AnalysisRecord

from backend.disaster_intelligence.service import DisasterIntelligenceService
from backend.disaster_intelligence.store import AnalysisStore
from backend.disaster_intelligence.analyzer import DisasterAnalyzer
from backend.disaster_intelligence.exceptions import (
    NoDataAvailable,
    ProviderUnavailable,
    RateLimitExceeded,
)

logger = logging.getLogger("crisismind.disaster_routes")

router = APIRouter(prefix="/disasters", tags=["Disasters"])

# Singleton service + store (lazily initialised)
_service: Optional[DisasterIntelligenceService] = None
_store: Optional[AnalysisStore] = None


def _get_service() -> DisasterIntelligenceService:
    global _service
    if _service is None:
        _service = DisasterIntelligenceService()
    return _service


def _get_store() -> AnalysisStore:
    global _store
    if _store is None:
        _store = AnalysisStore()
    return _store


def _get_analyzer(request: Request) -> DisasterAnalyzer:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise RuntimeError("Analysis engine not initialised.")
    return DisasterAnalyzer(engine, _get_store())


# ── Response schemas ──────────────────────────────────────────────────


class DisasterEventSummary(BaseModel):
    id: str
    title: str
    type: str
    country: str
    date: Optional[str] = None
    description: str
    source: str
    analysis_status: str = "READY_FOR_AI"


class DisasterEventDetail(BaseModel):
    id: str
    title: str
    type: str
    country: str
    date: Optional[str] = None
    description: str
    source: str
    source_url: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    image_url: Optional[str] = None
    severity: Optional[float] = None
    analysis_status: str = "READY_FOR_AI"
    metadata: dict = {}


class AnalyzeResponse(BaseModel):
    success: bool
    analysis_id: str
    event: dict
    analysis: Optional[dict] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    analysis_mode: Optional[str] = None
    processing_time_ms: float


# ── Routes ─────────────────────────────────────────────────────────────


@router.get("/events", response_model=list[DisasterEventSummary])
def list_events(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100, description="Max events to return"),
    disaster_type: Optional[str] = Query(default=None, description="Filter by disaster type (e.g. flood, earthquake)"),
    country: Optional[str] = Query(default=None, description="Filter by country name"),
):
    """Return a list of recent disaster events from the active provider."""
    service = _get_service()
    store = _get_store()

    try:
        result = service.get_events(
            disaster_type=disaster_type,
            country=country,
            limit=limit,
        )
    except ProviderUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except NoDataAvailable as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error fetching events")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    summaries = []
    for event in result.events:
        summaries.append(
            DisasterEventSummary(
                id=event.event_id,
                title=event.title,
                type=event.disaster_type,
                country=event.country,
                date=event.event_date.isoformat() if event.event_date else None,
                description=event.description,
                source=event.source,
                analysis_status=store.get_status(event.event_id),
            )
        )

    return summaries


@router.get("/events/{event_id}", response_model=DisasterEventDetail)
def get_event(event_id: str, request: Request):
    """Return full detail for a single disaster event."""
    service = _get_service()
    store = _get_store()

    try:
        event = service.get_event(event_id)
    except ProviderUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except NoDataAvailable as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error fetching event")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    return DisasterEventDetail(
        id=event.event_id,
        title=event.title,
        type=event.disaster_type,
        country=event.country,
        date=event.event_date.isoformat() if event.event_date else None,
        description=event.description,
        source=event.source,
        source_url=event.source_url,
        latitude=event.latitude,
        longitude=event.longitude,
        image_url=event.image_url,
        severity=event.severity,
        analysis_status=store.get_status(event.event_id),
        metadata=event.metadata,
    )


@router.post("/events/{event_id}/analyze", response_model=AnalyzeResponse)
def analyze_event(event_id: str, request: Request):
    """Run AI analysis on a disaster event and return an emergency assessment."""
    service = _get_service()
    analyzer = _get_analyzer(request)
    store = _get_store()

    # Check if already analysed (return cached)
    cached = store.get(event_id)
    if cached is not None:
        logger.info("Returning cached analysis for event=%s", event_id)
        return cached

    try:
        event = service.get_event(event_id)
    except (ProviderUnavailable, NoDataAvailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error fetching event %s", event_id)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    try:
        result = analyzer.analyze(event)
    except Exception as exc:
        logger.exception("Analysis failed for event %s", event_id)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Analysis returned no result"),
        )

    # Attach analysis_id to result
    result["analysis_id"] = event_id

    # Track in dashboard
    try:
        analysis = result.get("analysis", {})
        event_info = result.get("event", {})
        record = AnalysisRecord(
            record_id=event_id,
            record_type="live",
            source="disaster_intelligence",
            severity=analysis.get("severity"),
            confidence=analysis.get("confidence"),
            priority=analysis.get("priority"),
            disaster_type=event_info.get("type"),
            country=event_info.get("country"),
            analysis_mode=result.get("analysis_mode"),
            processing_time_ms=result.get("processing_time_ms", 0),
            event_title=event_info.get("title"),
            success=True,
            description=event.description if hasattr(event, 'description') else None,
            model_used=result.get("model_used"),
            infrastructure_impact=analysis.get("infrastructure_impact"),
            emergency_level=analysis.get("emergency_level"),
            recommended_actions=analysis.get("recommended_actions", []),
            raw_data=result,
        )
        dashboard_svc = getattr(request.app.state, "dashboard", None)
        if dashboard_svc:
            dashboard_svc.track_live(record)
    except Exception:
        logger.exception("Failed to track live analysis in dashboard")

    return result
