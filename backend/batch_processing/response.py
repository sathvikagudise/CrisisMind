from pydantic import BaseModel
from typing import Optional


class BatchFileResult(BaseModel):
    filename: str
    success: bool
    severity: Optional[str] = None
    confidence: Optional[float] = None
    priority: Optional[str] = None
    infrastructure_impact: Optional[str] = None
    emergency_level: Optional[str] = None
    recommended_actions: list[str] = []
    processing_time_ms: float = 0
    error: Optional[str] = None


class BatchSummary(BaseModel):
    total_files: int
    successful: int
    failed: int
    high_priority: int
    medium_priority: int
    low_priority: int
    avg_confidence: float
    avg_processing_time_ms: float
    severity_distribution: dict[str, int]
    priority_distribution: dict[str, int]


class BatchProgressResponse(BaseModel):
    batch_id: str
    status: str
    total_files: int
    completed: int
    failed: int
    pending: int
    progress_pct: float
    current_file: Optional[str] = None
    estimated_remaining_seconds: Optional[float] = None
    elapsed_seconds: float


class BatchResultsResponse(BaseModel):
    batch_id: str
    status: str
    total_files: int
    processing_time_seconds: float
    results: list[BatchFileResult]
    summary: Optional[BatchSummary] = None
