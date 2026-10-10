import asyncio
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.api.config import (
    MAX_IMAGE_SIZE_MB, ALLOWED_IMAGE_TYPES,
)
from backend.api.schemas import HealthResponse, ErrorResponse
from backend.api.disaster_routes import router as disaster_router
from backend.api.batch_routes import router as batch_router
from backend.api.dashboard_routes import router as dashboard_router
from backend.api.report_routes import router as report_router
from backend.api.explain_routes import router as explain_router
from backend.api.analysis_center_routes import router as analysis_center_router
from backend.analysis_center import AnalysisCenterService
from backend.analysis_engine import AnalysisEngine, AnalysisRequest, load_all_models
from backend.config.shared import CHECKPOINTS_DIR, IMG_CKPT, TEXT_CKPT, GATED_CKPT, DEVICE
from backend.disaster_intelligence.store import AnalysisStore
from backend.dashboard import DashboardStore, DashboardService, AnalysisRecord
from backend.reporting import ReportService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("crisismind")

app = FastAPI(
    title="CrisisMind API",
    description="Multimodal disaster damage severity classification using Gated Fusion (EfficientNet-B0 + DistilBERT).",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(disaster_router)
app.include_router(batch_router)
app.include_router(dashboard_router)
app.include_router(report_router)
app.include_router(explain_router)
app.include_router(analysis_center_router)

# Serve saved explanation images
from backend.explainability.utils import EXPLANATIONS_DIR
EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs/explanations", StaticFiles(directory=str(EXPLANATIONS_DIR)), name="explanations")

engine: AnalysisEngine | None = None


@app.on_event("startup")
def startup():
    global engine
    logger.info("Starting CrisisMind API...")

    # ── Log checkpoint paths ─────────────────────────────────────────────
    logger.info("Device: %s", DEVICE)
    logger.info("Checkpoints directory: %s", CHECKPOINTS_DIR)
    logger.info("  IMAGE model  → %s  [exists=%s]", IMG_CKPT, IMG_CKPT.exists())
    logger.info("  TEXT  model  → %s  [exists=%s]", TEXT_CKPT, TEXT_CKPT.exists())
    logger.info("  GATED model  → %s  [exists=%s]", GATED_CKPT, GATED_CKPT.exists())

    models = load_all_models()
    engine = AnalysisEngine(models)
    app.state.engine = engine
    app.state.models_loaded = len(models)
    app.state.store = AnalysisStore()
    app.state.dashboard_store = DashboardStore()
    app.state.dashboard = DashboardService(app.state.dashboard_store)
    app.state.report_service = ReportService(app.state.dashboard_store)
    app.state.analysis_center = AnalysisCenterService(app.state.dashboard_store, app.state.store)
    logger.info("CrisisMind API ready — AnalysisEngine + AnalysisStore + Dashboard + Reports + AnalysisCenter initialised.")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(status="healthy", models_loaded=getattr(app.state, "models_loaded", 0))


def _get_engine() -> AnalysisEngine:
    if engine is None:
        raise RuntimeError("Engine not initialised. Server may still be starting.")
    return engine


@app.post(
    "/predict",
    response_model=None,
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        413: {"description": "Image too large"},
        422: {"description": "Unsupported image format"},
    },
)
async def predict(
    image: UploadFile | None = File(None),
    tweet_text: str | None = Form(None),
):
    pil_image = None
    text_value = tweet_text.strip() if tweet_text else None

    if not image and not text_value:
        raise HTTPException(
            status_code=400,
            detail="Provide an image, tweet_text, or both.",
        )

    if image and image.filename:
        content_type = image.content_type or ""
        if content_type and content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported image type '{content_type}'. Allowed: {ALLOWED_IMAGE_TYPES}",
            )
        contents = await image.read()
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large ({size_mb:.1f} MB). Max: {MAX_IMAGE_SIZE_MB} MB",
            )
        try:
            pil_image = Image.open(io.BytesIO(contents))
            pil_image.verify()
            pil_image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid image file")

    eng = _get_engine()
    request = AnalysisRequest(
        image=pil_image,
        text=text_value,
        source="manual_upload",
    )

    result = await asyncio.to_thread(eng.analyze, request)

    if not result["success"]:
        err_type = result.get("error_type", "unknown")
        status = 400 if err_type == "validation" else 500
        raise HTTPException(status_code=status, detail=result.get("error", "Analysis failed"))

    # Track in dashboard
    try:
        decision = result.get("decision", {})
        record = AnalysisRecord(
            record_id=result.get("analysis_id", ""),
            record_type="manual",
            source="manual_upload",
            severity=result.get("prediction", {}).get("severity"),
            confidence=result.get("prediction", {}).get("confidence"),
            priority=decision.get("priority"),
            analysis_mode=result.get("analysis_mode"),
            processing_time_ms=result.get("processing_time_ms", 0),
            success=True,
            description=(tweet_text or "").strip() or None,
            model_used=result.get("model_used") or result.get("model"),
            infrastructure_impact=decision.get("infrastructure_impact"),
            emergency_level=decision.get("emergency_level"),
            recommended_actions=decision.get("recommended_actions", []),
            raw_data=result,
        )
        app.state.dashboard.track_manual(record)
    except Exception:
        logger.exception("Failed to track manual analysis in dashboard")

    return result
