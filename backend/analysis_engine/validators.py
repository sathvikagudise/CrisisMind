import logging

from backend.analysis_engine.request import AnalysisRequest, AnalysisMode
from backend.analysis_engine.exceptions import ValidationError

logger = logging.getLogger("crisismind")

MIN_TEXT_LENGTH = 1
MAX_TEXT_LENGTH = 1000


def validate_request(request: AnalysisRequest) -> AnalysisMode:
    has_image = request.image is not None
    has_text = request.text is not None and len(request.text.strip()) > 0

    if request.analysis_mode:
        mode = request.analysis_mode
    elif has_image and has_text:
        mode = AnalysisMode.MULTIMODAL
    elif has_image:
        mode = AnalysisMode.IMAGE_ONLY
    elif has_text:
        mode = AnalysisMode.TEXT_ONLY
    else:
        raise ValidationError(
            "No input provided. Supply an image, text, or both."
        )

    if mode in (AnalysisMode.IMAGE_ONLY, AnalysisMode.MULTIMODAL) and not has_image:
        raise ValidationError("Image is required for the selected analysis mode.")
    if mode in (AnalysisMode.TEXT_ONLY, AnalysisMode.MULTIMODAL) and not has_text:
        raise ValidationError("Text is required for the selected analysis mode.")

    if has_text and len(request.text.strip()) > MAX_TEXT_LENGTH:
        raise ValidationError(
            f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters."
        )

    logger.info("Request validated: mode=%s, source=%s", mode.value, request.source)
    return mode
