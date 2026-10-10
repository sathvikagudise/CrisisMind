# Analysis Center

Unified interface for browsing, searching, filtering, and exporting all completed analyses — manual, live, and batch — in one place.

## Architecture

```
backend/analysis_center/
├── __init__.py          # exports AnalysisCenterService
├── service.py           # main service: aggregates DashboardStore + AnalysisStore
├── filters.py           # filter predicates (type, severity, priority, date, …)
├── search.py            # full-text search across multiple fields
├── pagination.py        # paginate helper
└── README.md
```

## How it works

The `AnalysisCenterService` pulls records from two existing stores:

1. **`DashboardStore`** (`.`)\
   Contains `AnalysisRecord` objects for manual uploads and batch processing results.

2. **`AnalysisStore`** (`.`)\
   Contains live disaster event analysis results keyed by event ID.

Each source is normalised into a unified dict shape and combined into a single sorted list (newest first by default). Search, filters, sort, and pagination are applied on top of this unified list.

## API Endpoints

All endpoints are under the `/analysis-center` prefix:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analysis-center` | Paginated list with filters, sort, search |
| GET | `/analysis-center/search?q=...` | Full-text search |
| GET | `/analysis-center/{analysis_id}` | Single record detail |
| GET | `/analysis-center/export` | CSV export of (optionally filtered) results |
| GET | `/analysis-center/statistics` | Aggregate statistics |

### Query parameters for `GET /analysis-center`

- `page` (int, default 1)
- `per_page` (int, default 20, max 100)
- `sort_by` — `newest` / `oldest` / `highest_confidence` / `lowest_confidence` / `highest_priority`
- `query` — full-text search
- `record_type` — `manual` / `live` / `batch`
- `analysis_mode`
- `severity`
- `priority`
- `disaster_type`
- `source`
- `date_from` — ISO date string (inclusive lower bound)
- `date_to` — ISO date string (inclusive upper bound)

## Normalised Record Shape

```python
{
    "analysis_id": str,
    "record_type": str,        # "manual" | "live" | "batch"
    "timestamp": str,           # ISO 8601
    "source": str,
    "severity": str | None,
    "confidence": float | None,
    "priority": str | None,
    "analysis_mode": str | None,
    "disaster_type": str | None,
    "processing_time_ms": float,
    "filename": str | None,
    "event_title": str | None,
    "batch_id": str | None,
    "description": str | None,
    "country": str | None,
    "model_used": str | None,
    "infrastructure_impact": str | None,
    "emergency_level": str | None,
    "recommended_actions": list[str],
    "success": bool,
    "error": str | None,
    "has_explanation": bool,    # whether a Grad-CAM overlay exists
}
```

## Frontend

A dedicated **Analysis Center** page is available at `/analysis-center` with:

- Search bar (full-text across ID, disaster type, country, description, source, event title, filename)
- Filter panel (type, severity, priority)
- Sort control (newest, oldest, highest/lowest confidence, highest priority)
- Statistics summary cards (totals by type, avg confidence, avg time)
- Paginated data table with type badge, severity, confidence, priority
- Detail modal with full record information, recommended actions, report download
- CSV export button

Report PDF downloads reuse the existing `/report/manual/{id}` and `/report/live/{id}` URLs.
