"""Explainability API routes — Grad-CAM generation."""

import io
import logging

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from starlette.responses import Response
from PIL import Image

from backend.api.config import ALLOWED_IMAGE_TYPES, MAX_IMAGE_SIZE_MB
from backend.explainability.service import generate_explanation
from backend.explainability.utils import EXPLANATIONS_DIR

logger = logging.getLogger("crisismind.api.explain")

router = APIRouter(prefix="/explainability", tags=["Explainability"])


@router.post("/gradcam")
async def gradcam_endpoint(
    request: Request,
    image: UploadFile = File(...),
    tweet_text: str = Form(None),
    analysis_id: str = Form(None),
):
    """Generate Grad-CAM explanation for an image.

    - ``image``: JPEG/PNG/WebP/BMP (required).
    - ``tweet_text``: Optional text (if provided, uses multimodal pipeline).
    - ``analysis_id``: Optional ID to associate the explanation with a prior analysis.
    """
    # Validate image
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

    # Get analysis engine from app state
    eng = getattr(request.app.state, "engine", None)
    if eng is None:
        raise HTTPException(status_code=503, detail="Analysis engine not ready")

    try:
        text_val = tweet_text.strip() if tweet_text else None
        result = generate_explanation(
            engine=eng,
            image=pil_image,
            tweet_text=text_val,
            analysis_id=analysis_id or None,
        )
        return result
    except Exception as e:
        logger.exception("Grad-CAM generation failed")
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")


# Serve explanation images as static files
@router.get("/outputs/{analysis_id}/{filename}")
async def serve_explanation_image(analysis_id: str, filename: str):
    """Serve saved explanation images."""
    import os
    filepath = EXPLANATIONS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="Explanation image not found")
    data = filepath.read_bytes()
    ext = filepath.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")
    return Response(content=data, media_type=media_type)
