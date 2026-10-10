"""Central Analysis Engine — the only entry point for all AI inference."""

import uuid
import time
import logging
import torch

from PIL import Image

from backend.analysis_engine.request import AnalysisRequest, AnalysisMode
from backend.analysis_engine.validators import validate_request
from backend.analysis_engine.model_selector import select_model
from backend.analysis_engine.response_builder import build_response, build_error_response
from backend.analysis_engine.exceptions import AnalysisError, ValidationError
from backend.api.preprocessing import preprocess_image, tokenize_text
from backend.api.config import DEVICE, SEVERITY_MAP
from backend.api.schemas import PredictionResult

logger = logging.getLogger("crisismind")


class AnalysisEngine:
    """Unified inference engine.

    Usage::

        engine = AnalysisEngine(model_dict)
        result = engine.analyze(request)
    """

    def __init__(self, models: dict):
        self.models = models

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self, request: AnalysisRequest) -> dict:
        """Validate → determine mode → select model → infer → respond."""
        start = time.perf_counter()
        analysis_id = request.request_id or str(uuid.uuid4())
        logger.info("Request received: id=%s source=%s", analysis_id, request.source)

        try:
            mode = validate_request(request)
            logger.info("Analysis mode determined: %s", mode.value)

            model_name = select_model(mode)

            if mode == AnalysisMode.MULTIMODAL:
                prediction = self._run_multimodal(request.image, request.text)
            elif mode == AnalysisMode.IMAGE_ONLY:
                prediction = self._run_image_only(request.image)
            else:
                prediction = self._run_text_only(request.text)

            elapsed = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "Inference completed: id=%s mode=%s severity=%s time=%.1fms",
                analysis_id, mode.value, prediction.severity, elapsed,
            )

            return build_response(
                analysis_id=analysis_id,
                analysis_mode=mode,
                prediction=prediction,
                model_used=model_name,
                processing_time_ms=elapsed,
                metadata=request.metadata,
            )

        except ValidationError as e:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            logger.warning("Validation error: id=%s error=%s", analysis_id, e)
            return build_error_response(analysis_id, str(e), elapsed, error_type="validation")

        except AnalysisError as e:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            logger.error("Analysis error: id=%s error=%s", analysis_id, e)
            return build_error_response(analysis_id, str(e), elapsed, error_type="analysis")

        except Exception as e:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            logger.exception("Unexpected error: id=%s", analysis_id)
            return build_error_response(analysis_id, f"Internal error: {e}", elapsed, error_type="internal")

    # ── Inference helpers ───────────────────────────────────────────────

    def _run_multimodal(self, image: Image.Image, text: str) -> PredictionResult:
        img_tensor = preprocess_image(image).to(DEVICE)
        input_ids, attention_mask = tokenize_text(text)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        img_encoder = self.models["img_encoder"]
        txt_encoder = self.models["txt_encoder"]
        gated_model = self.models["gated_model"]

        with torch.no_grad():
            img_feat = img_encoder(img_tensor)
            txt_feat = txt_encoder(input_ids, attention_mask)
            logits, gates = gated_model(img_feat, txt_feat)

        logger.info("Gates: img=%.3f txt=%.3f", gates[0, 0].item(), gates[0, 1].item())
        return self._prediction_from_logits(logits)

    def _run_image_only(self, image: Image.Image) -> PredictionResult:
        img_tensor = preprocess_image(image).to(DEVICE)
        classifier = self.models["image_classifier"]

        with torch.no_grad():
            logits = classifier(img_tensor)

        return self._prediction_from_logits(logits)

    def _run_text_only(self, text: str) -> PredictionResult:
        input_ids, attention_mask = tokenize_text(text)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        classifier = self.models["text_classifier"]

        with torch.no_grad():
            outputs = classifier(input_ids, attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        return self._prediction_from_logits(logits)

    @staticmethod
    def _prediction_from_logits(logits: torch.Tensor) -> PredictionResult:
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item() * 100)
        severity = SEVERITY_MAP[pred_idx]
        return PredictionResult(severity=severity, confidence=round(confidence, 2))
