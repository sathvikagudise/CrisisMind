from pydantic import BaseModel, Field
from typing import Optional


class BatchUploadResponse(BaseModel):
    batch_id: str
    total_files: int
    has_csv: bool
    message: str


class BatchStartRequest(BaseModel):
    batch_id: str


class BatchStatusResponse(BaseModel):
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
