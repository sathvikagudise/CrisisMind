import time
import logging
from typing import Optional

logger = logging.getLogger("crisismind")


class ProgressTracker:
    def __init__(self, batch_id: str, total: int):
        self.batch_id = batch_id
        self.total = total
        self.completed = 0
        self.failed = 0
        self.current_file: Optional[str] = None
        self.start_time: Optional[float] = None
        self.status = "pending"
        self.results: list = []

    def start(self):
        self.start_time = time.perf_counter()
        self.status = "processing"
        logger.info("Batch %s started — %d files", self.batch_id, self.total)

    def complete_one(self, filename: str, result: dict):
        self.completed += 1
        if not result.get("success", True):
            self.failed += 1
        self.results.append(result)
        logger.info(
            "Batch %s: %d/%d — %s (failed=%d)",
            self.batch_id, self.completed, self.total, filename, self.failed,
        )

    def finish(self):
        self.status = "completed" if self.failed < self.total else "failed"
        elapsed = self.elapsed
        logger.info(
            "Batch %s finished: status=%s completed=%d failed=%d time=%.2fs",
            self.batch_id, self.status, self.completed, self.failed, elapsed,
        )

    @property
    def pending(self) -> int:
        return self.total - self.completed

    @property
    def progress_pct(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.completed / self.total) * 100

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    @property
    def estimated_remaining(self) -> Optional[float]:
        if self.completed == 0 or self.start_time is None:
            return None
        rate = self.completed / self.elapsed
        return (self.pending / rate) if rate > 0 else None
