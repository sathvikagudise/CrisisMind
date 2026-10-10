import io
import csv
import logging
from typing import Optional

from PIL import Image

from backend.api.config import ALLOWED_IMAGE_TYPES

logger = logging.getLogger("crisismind")


def validate_image_file(filename: str, contents: bytes) -> Optional[str]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                 "webp": "image/webp", "bmp": "image/bmp"}
    mime = mime_map.get(ext)
    if mime not in ALLOWED_IMAGE_TYPES:
        return f"Unsupported image format '{ext}'"

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        return f"Invalid or corrupted image file: {filename}"

    return None


def parse_csv_contents(contents: bytes) -> Optional[list[dict]]:
    try:
        text = contents.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return None
        required = {"filename"}
        if not required.issubset(reader.fieldnames or set()):
            logger.warning("CSV missing required 'filename' column; found: %s", reader.fieldnames)
            return None
        return rows
    except Exception as e:
        logger.warning("Failed to parse CSV: %s", e)
        return None
