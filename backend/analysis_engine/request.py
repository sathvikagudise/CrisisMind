from enum import Enum
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class AnalysisMode(str, Enum):
    IMAGE_ONLY = "IMAGE_ONLY"
    TEXT_ONLY = "TEXT_ONLY"
    MULTIMODAL = "MULTIMODAL"


class AnalysisRequest(BaseModel):
    image: Optional[Any] = None
    text: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "manual_upload"
    analysis_mode: Optional[AnalysisMode] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
