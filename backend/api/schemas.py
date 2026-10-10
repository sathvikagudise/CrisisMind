from pydantic import BaseModel


class PredictionResult(BaseModel):
    severity: str
    confidence: float


class PredictResponse(BaseModel):
    success: bool
    prediction: PredictionResult
    model: str
    processing_time_ms: float


class ErrorResponse(BaseModel):
    success: bool
    error: str
    model: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: int = 0
