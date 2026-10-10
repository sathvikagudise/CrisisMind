import asyncio
import io
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import Response, PlainTextResponse

from backend.api.config import MAX_IMAGE_SIZE_MB
from backend.batch_processing import (
    BatchUploadResponse,
    BatchResultsResponse,
    BatchFileResult,
    BatchProcessor,
    validate_image_file,
    parse_csv_contents,
    export_csv,
    export_pdf_text,
    compute_summary,
)
from backend.dashboard import AnalysisRecord

logger = logging.getLogger("crisismind")

router = APIRouter(prefix="/batch", tags=["Batch Processing"])

_batch_processor: BatchProcessor | None = None


def _make_tracker_callback(dashboard_svc):
    """Return a callback that tracks each batch file result in the dashboard service."""
    def callback(result: dict, batch_id: str):
        record = AnalysisRecord(
            record_id=f"{batch_id}-{result.get('filename', 'unknown')}",
            record_type="batch",
            source="batch_processing",
            severity=result.get("severity"),
            confidence=result.get("confidence"),
            priority=result.get("priority"),
            analysis_mode="IMAGE_ONLY",
            processing_time_ms=result.get("processing_time_ms", 0),
            filename=result.get("filename"),
            batch_id=batch_id,
            success=result.get("success", False),
            error=result.get("error"),
            infrastructure_impact=result.get("infrastructure_impact"),
            emergency_level=result.get("emergency_level"),
            recommended_actions=result.get("recommended_actions", []),
            raw_data=result,
        )
        dashboard_svc.track_batch(record)
    return callback


def get_processor(request: Request = None) -> BatchProcessor:
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    if request is not None:
        dashboard_svc = getattr(request.app.state, "dashboard", None)
        if dashboard_svc:
            _batch_processor._on_file_complete = _make_tracker_callback(dashboard_svc)
    return _batch_processor


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (or mixed image+text files)"),
    csv_file: Optional[UploadFile] = File(None, description="Optional CSV with filename,text columns"),
):
    processor = get_processor(request)

    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    batch_id = str(uuid.uuid4())[:8]
    file_data: dict[str, bytes] = {}
    errors: list[str] = []
    total_size_mb = 0

    for f in files:
        if not f.filename:
            continue
        contents = await f.read()
        size_mb = len(contents) / (1024 * 1024)
        total_size_mb += size_mb

        if size_mb > MAX_IMAGE_SIZE_MB:
            errors.append(f"'{f.filename}' exceeds size limit ({size_mb:.1f} MB > {MAX_IMAGE_SIZE_MB} MB)")
            continue

        err = validate_image_file(f.filename, contents)
        if err:
            errors.append(f"'{f.filename}': {err}")
            continue

        file_data[f.filename] = contents

    if not file_data:
        detail = "No valid files uploaded"
        if errors:
            detail += ": " + "; ".join(errors[:3])
        raise HTTPException(status_code=422, detail=detail)

    csv_rows = None
    if csv_file and csv_file.filename:
        csv_contents = await csv_file.read()
        csv_rows = parse_csv_contents(csv_contents)
        if csv_rows is None:
            logger.warning("Batch %s: CSV upload could not be parsed or missing 'filename' column", batch_id)

    processor.create_batch(batch_id, file_data, csv_rows)

    msg = f"Batch {batch_id} created with {len(file_data)} file(s)"
    if errors:
        msg += f". {len(errors)} file(s) skipped due to errors."
    if csv_rows is not None:
        msg += f". CSV with {len(csv_rows)} row(s) loaded."

    return BatchUploadResponse(
        batch_id=batch_id,
        total_files=len(file_data),
        has_csv=csv_rows is not None,
        message=msg,
    )


@router.post("/start/{batch_id}", response_model=dict)
async def start_batch(batch_id: str, request: Request):
    processor = get_processor(request)
    tracker = processor.get_tracker(batch_id)
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    if tracker.status == "processing":
        raise HTTPException(status_code=409, detail=f"Batch '{batch_id}' is already processing")

    if tracker.status == "completed":
        raise HTTPException(status_code=409, detail=f"Batch '{batch_id}' is already completed")

    engine = request.app.state.engine
    loop = asyncio.get_event_loop()
    loop.create_task(processor.process_batch(batch_id, engine))

    return {
        "batch_id": batch_id,
        "status": "processing",
        "message": f"Batch {batch_id} processing started",
    }


@router.get("/status/{batch_id}")
async def batch_status(batch_id: str):
    processor = get_processor()
    tracker = processor.get_tracker(batch_id)
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    return {
        "batch_id": batch_id,
        "status": tracker.status,
        "total_files": tracker.total,
        "completed": tracker.completed,
        "failed": tracker.failed,
        "pending": tracker.pending,
        "progress_pct": round(tracker.progress_pct, 1),
        "current_file": tracker.current_file,
        "estimated_remaining_seconds": (
            round(tracker.estimated_remaining, 1) if tracker.estimated_remaining is not None else None
        ),
        "elapsed_seconds": round(tracker.elapsed, 1),
    }


@router.get("/results/{batch_id}", response_model=BatchResultsResponse)
async def batch_results(batch_id: str):
    processor = get_processor()
    tracker = processor.get_tracker(batch_id)
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    results_raw = processor.get_results(batch_id) or []
    results = [BatchFileResult(**r) for r in results_raw]
    summary = compute_summary(results) if results else None

    return BatchResultsResponse(
        batch_id=batch_id,
        status=tracker.status,
        total_files=tracker.total,
        processing_time_seconds=round(tracker.elapsed, 2),
        results=results,
        summary=summary,
    )


@router.get("/download/csv/{batch_id}")
async def download_csv(batch_id: str):
    processor = get_processor()
    tracker = processor.get_tracker(batch_id)
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    results_raw = processor.get_results(batch_id) or []
    results = [BatchFileResult(**r) for r in results_raw]
    summary = compute_summary(results) if results else None

    csv_content = export_csv(results, summary)

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=crisismind_batch_{batch_id}.csv",
        },
    )


@router.get("/download/pdf/{batch_id}")
async def download_pdf(batch_id: str):
    processor = get_processor()
    tracker = processor.get_tracker(batch_id)
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    results_raw = processor.get_results(batch_id) or []
    results = [BatchFileResult(**r) for r in results_raw]
    summary = compute_summary(results) if results else None

    pdf_text = export_pdf_text(results, summary)

    return PlainTextResponse(
        content=pdf_text,
        headers={
            "Content-Disposition": f"attachment; filename=crisismind_batch_{batch_id}.txt",
        },
    )
