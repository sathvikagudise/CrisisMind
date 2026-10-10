"""Pagination helpers for the Analysis Center."""

import math


def paginate(items: list, page: int = 1, per_page: int = 20) -> dict:
    """Return a paginated slice of *items* with metadata.

    Returns::

        {
            "items": [...],
            "page": 1,
            "per_page": 20,
            "total_items": 100,
            "total_pages": 5,
            "has_next": True,
            "has_prev": False,
        }
    """
    total = len(items)
    total_pages = max(1, math.ceil(total / per_page))
    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    end = start + per_page
    sliced = items[start:end]

    return {
        "items": sliced,
        "page": page,
        "per_page": per_page,
        "total_items": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }
