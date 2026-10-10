import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query

logger = logging.getLogger("crisismind.dashboard_routes")

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


def _get_service(request: Request):
    svc = getattr(request.app.state, "dashboard", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Dashboard service not initialised")
    return svc


@router.get("/summary")
def dashboard_summary(request: Request):
    svc = _get_service(request)
    return svc.get_summary()


@router.get("/statistics")
def dashboard_statistics(request: Request):
    svc = _get_service(request)
    return svc.get_statistics()


@router.get("/charts")
def dashboard_charts(request: Request):
    svc = _get_service(request)
    return svc.get_charts()


@router.get("/timeline")
def dashboard_timeline(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
):
    svc = _get_service(request)
    return svc.get_timeline(limit)


@router.get("/recent")
def dashboard_recent(
    request: Request,
    limit: int = Query(default=10, ge=1, le=50),
):
    svc = _get_service(request)
    return svc.get_recent(limit)
