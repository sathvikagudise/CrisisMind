"""API routes for the Analysis Center."""

import logging
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from backend.analysis_center import AnalysisCenterService

logger = logging.getLogger("crisismind.analysis_center.api")
router = APIRouter(prefix="/analysis-center", tags=["Analysis Center"])


def _get_service(request: Request) -> AnalysisCenterService:
    svc: AnalysisCenterService | None = request.app.state.analysis_center
    if svc is None:
        raise HTTPException(status_code=503, detail="Analysis Center not ready")
    return svc


@router.get("")
def list_analyses(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_by: str = Query("newest", regex="^(newest|oldest|highest_confidence|lowest_confidence|highest_priority)$"),
    query: str = Query(None, description="Full-text search"),
    record_type: str = Query(None, regex="^(manual|live|batch)$"),
    analysis_mode: str = Query(None),
    severity: str = Query(None),
    priority: str = Query(None),
    disaster_type: str = Query(None),
    source: str = Query(None),
    date_from: str = Query(None, description="ISO date (inclusive lower bound)"),
    date_to: str = Query(None, description="ISO date (inclusive upper bound)"),
):
    svc = _get_service(request)
    return svc.list_analyses(
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        query=query,
        record_type=record_type,
        analysis_mode=analysis_mode,
        severity=severity,
        priority=priority,
        disaster_type=disaster_type,
        source=source,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/search")
def search_analyses(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_by: str = Query("newest"),
):
    svc = _get_service(request)
    return svc.search(q, page=page, per_page=per_page, sort_by=sort_by)


@router.get("/export")
def export_csv(
    request: Request,
    query: str = Query(None),
    record_type: str = Query(None),
    analysis_mode: str = Query(None),
    severity: str = Query(None),
    priority: str = Query(None),
    disaster_type: str = Query(None),
    source: str = Query(None),
    date_from: str = Query(None),
    date_to: str = Query(None),
):
    svc = _get_service(request)
    csv_content = svc.export_csv(
        query=query,
        record_type=record_type,
        analysis_mode=analysis_mode,
        severity=severity,
        priority=priority,
        disaster_type=disaster_type,
        source=source,
        date_from=date_from,
        date_to=date_to,
    )
    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=analysis_center_export.csv"},
    )


@router.get("/statistics")
def get_statistics(request: Request):
    svc = _get_service(request)
    return svc.get_statistics()


@router.get("/{analysis_id}")
def get_analysis(request: Request, analysis_id: str):
    svc = _get_service(request)
    record = svc.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record
