import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend.reporting import ReportService

logger = logging.getLogger("crisismind.report_routes")

router = APIRouter(prefix="/report", tags=["Reports"])


def _get_service(request: Request) -> ReportService:
    svc = getattr(request.app.state, "report_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Report service not initialised")
    return svc


@router.get("/manual/{analysis_id}")
def report_manual(analysis_id: str, request: Request):
    svc = _get_service(request)
    try:
        pdf_bytes = svc.generate_manual_report(analysis_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to generate manual report")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=crisismind_manual_{analysis_id}.pdf",
        },
    )


@router.get("/live/{analysis_id}")
def report_live(analysis_id: str, request: Request):
    svc = _get_service(request)
    try:
        pdf_bytes = svc.generate_live_report(analysis_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to generate live report")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=crisismind_live_{analysis_id}.pdf",
        },
    )


@router.get("/batch/{batch_id}")
def report_batch(batch_id: str, request: Request):
    svc = _get_service(request)
    try:
        pdf_bytes = svc.generate_batch_report(batch_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to generate batch report")
        raise HTTPException(status_code=500, detail=f"Batch report generation failed: {e}")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=crisismind_batch_{batch_id}.pdf",
        },
    )


@router.get("/csv/{batch_id}")
def report_csv(batch_id: str, request: Request):
    svc = _get_service(request)
    try:
        csv_content = svc.generate_batch_csv(batch_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to generate batch CSV")
        raise HTTPException(status_code=500, detail=f"CSV generation failed: {e}")
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=crisismind_batch_{batch_id}.csv",
        },
    )
