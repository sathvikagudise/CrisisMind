import logging

from backend.analysis_engine.request import AnalysisMode

logger = logging.getLogger("crisismind")

MODEL_NAMES = {
    AnalysisMode.IMAGE_ONLY: "EfficientNet-B0",
    AnalysisMode.TEXT_ONLY: "DistilBERT",
    AnalysisMode.MULTIMODAL: "GatedFusion v3",
}


def select_model(mode: AnalysisMode) -> str:
    name = MODEL_NAMES.get(mode)
    if name is None:
        name = "Unknown"
        logger.warning("No model mapping for mode %s", mode.value)
    logger.info("Model selected: %s (mode=%s)", name, mode.value)
    return name
