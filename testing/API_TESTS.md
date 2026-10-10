# CrisisMind — API Test Guide

> **Regenerated from source.** Verified against `backend/api/main.py` and all
> `*_routes.py` files as of 2026-07-09. Total: **27 endpoints** (15 GET, 2 POST direct + 5 router POST + 5 router GET = 27).

Start the backend:

```bash
cd backend
uvicorn backend.api.main:app --reload --port 8000
```

Base URL: `http://localhost:8000`

---

## System

### GET /health

| Field | Value |
|---|---|
| Method | GET |
| URL | `/health` |
| Parameters | None |

**Expected Response:** 200

```json
{
  "status": "healthy"
}
```

---

## Prediction

### POST /predict

| Field | Value |
|---|---|
| Method | POST |
| URL | `/predict` |
| Content-Type | `multipart/form-data` |
| Parameters | `image`: file (optional), `tweet_text`: string (optional) |

At least one of `image` or `tweet_text` must be provided.

**Input (image-only):**
- `image`: file from `testing/manual_test/` or `testing/sample_images/`

**Input (text-only):**
- `tweet_text`: description from `testing/manual_text_samples.md`

**Input (multimodal):**
- `image`: file
- `tweet_text`: description

**Expected Response:** 200

```json
{
  "analysis_id": "uuid-string",
  "analysis": {
    "severity": 72.5,
    "confidence": 85.3,
    "priority": "high",
    "disaster_type": "flood",
    "infrastructure_impact": { "roads": "moderate", "buildings": "severe" },
    "emergency_level": "high",
    "recommended_actions": [
      "Evacuate residents from low-lying areas.",
      "Deploy emergency shelters."
    ]
  },
  "analysis_mode": "IMAGE_ONLY",
  "processing_time_ms": 2340,
  "model_used": "efficientnet_b0_distilbert_gated_fusion_v1"
}
```

---

## Batch Processing

### POST /batch/upload

| Field | Value |
|---|---|
| Method | POST |
| URL | `/batch/upload` |
| Content-Type | `multipart/form-data` |
| Parameters | `files`: list of image files (required), `csv_file`: CSV file (optional) |

Upload images plus an optional CSV (`testing/sample_batch.csv`). Returns a batch ID — does **not** start processing.

**Input:**
- `files`: one or more JPEG/PNG files from `testing/sample_images/`
- `csv_file` (optional): `testing/sample_batch.csv`

**Expected Response:** 200

```json
{
  "batch_id": "a1b2c3d4",
  "total_files": 21,
  "has_csv": true,
  "message": "Batch a1b2c3d4 created with 21 file(s). CSV with 20 row(s) loaded."
}
```

---

### POST /batch/start/{batch_id}

| Field | Value |
|---|---|
| Method | POST |
| URL | `/batch/start/{batch_id}` |
| Parameters | `batch_id`: path (required) |

Starts processing for an uploaded batch. Must be called after `/batch/upload`.

**Expected Response:** 200

```json
{
  "batch_id": "a1b2c3d4",
  "status": "processing",
  "message": "Batch a1b2c3d4 processing started"
}
```

**Error responses:**
- 404 — batch not found
- 409 — already processing or completed

---

### GET /batch/status/{batch_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/batch/status/{batch_id}` |
| Parameters | `batch_id`: path (required) |

**Expected Response:** 200

```json
{
  "batch_id": "a1b2c3d4",
  "status": "completed",
  "total_files": 21,
  "completed": 21,
  "failed": 0,
  "pending": 0,
  "progress_pct": 100.0,
  "current_file": null,
  "estimated_remaining_seconds": null,
  "elapsed_seconds": 45.2
}
```

---

### GET /batch/results/{batch_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/batch/results/{batch_id}` |
| Parameters | `batch_id`: path (required) |

**Expected Response:** 200

```json
{
  "batch_id": "a1b2c3d4",
  "status": "completed",
  "total_files": 21,
  "processing_time_seconds": 45.2,
  "results": [
    {
      "filename": "906984251538776071_0.jpg",
      "success": true,
      "severity": 15.2,
      "confidence": 78.5,
      "priority": "low",
      "disaster_type": "flood",
      "infrastructure_impact": {},
      "emergency_level": "low",
      "recommended_actions": [],
      "processing_time_ms": 2100
    }
  ],
  "summary": {
    "total": 21,
    "successful": 21,
    "failed": 0,
    "avg_severity": 52.3,
    "avg_confidence": 81.2,
    "by_priority": { "low": 8, "medium": 7, "high": 4, "critical": 2 }
  }
}
```

---

### GET /batch/download/csv/{batch_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/batch/download/csv/{batch_id}` |
| Parameters | `batch_id`: path (required) |

**Expected Response:** 200 — `text/csv` file download

```
Content-Disposition: attachment; filename=crisismind_batch_a1b2c3d4.csv
```

---

### GET /batch/download/pdf/{batch_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/batch/download/pdf/{batch_id}` |
| Parameters | `batch_id`: path (required) |

**Expected Response:** 200 — plain text report download

```
Content-Disposition: attachment; filename=crisismind_batch_a1b2c3d4.txt
```

---

## Disaster Intelligence

### GET /disasters/events

| Field | Value |
|---|---|
| Method | GET |
| URL | `/disasters/events` |
| Parameters | `limit`: int 1–100 (default 20), `disaster_type`: string (optional), `country`: string (optional) |

**Expected Response:** 200

```json
[
  {
    "event_id": "evt-001",
    "title": "Hurricane Harvey — Downtown Flooding",
    "type": "flood",
    "location": "Houston, TX",
    "country": "USA",
    "severity": 0.7,
    "timestamp": "2026-07-09T12:00:00Z"
  }
]
```

---

### GET /disasters/events/{event_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/disasters/events/{event_id}` |
| Parameters | `event_id`: path (required) |

**Expected Response:** 200

```json
{
  "event_id": "evt-001",
  "title": "Hurricane Harvey — Downtown Flooding",
  "type": "flood",
  "location": "Houston, TX",
  "country": "USA",
  "severity": 0.7,
  "timestamp": "2026-07-09T12:00:00Z",
  "analyzed": true,
  "analysis": { "severity": 75.0, "confidence": 82.1, ... }
}
```

---

### POST /disasters/events/{event_id}/analyze

| Field | Value |
|---|---|
| Method | POST |
| URL | `/disasters/events/{event_id}/analyze` |
| Parameters | `event_id`: path (required) |

Trigger analysis for an existing event. The event must already exist in the store. Results are stored in the AnalysisStore.

**Expected Response:** 200

```json
{
  "event_id": "evt-001",
  "analysis": {
    "severity": 75.0,
    "confidence": 82.1,
    "priority": "high",
    "disaster_type": "flood",
    "infrastructure_impact": { "roads": "severe" },
    "emergency_level": "high",
    "recommended_actions": ["Evacuate immediately."]
  },
  "analysis_mode": "MULTIMODAL",
  "processing_time_ms": 3200,
  "model_used": "efficientnet_b0_distilbert_gated_fusion_v1"
}
```

**Error responses:**
- 404 — event not found

---

## Dashboard

### GET /dashboard/summary

| Field | Value |
|---|---|
| Method | GET |
| URL | `/dashboard/summary` |
| Parameters | None |

**Expected Response:** 200

```json
{
  "total_analyses": 42,
  "total_batches": 5,
  "total_disasters": 3,
  "by_type": { "manual": 10, "batch": 30, "live": 2 },
  "by_severity": { "little": 15, "mild": 12, "severe": 15 },
  "by_priority": { "low": 10, "medium": 15, "high": 12, "critical": 5 },
  "by_disaster_type": { "flood": 20, "earthquake": 10, "wildfire": 12 },
  "avg_confidence": 79.4,
  "avg_severity": 58.2
}
```

---

### GET /dashboard/statistics

| Field | Value |
|---|---|
| Method | GET |
| URL | `/dashboard/statistics` |
| Parameters | None |

**Expected Response:** 200 — extended statistics (counts, averages, distributions)

---

### GET /dashboard/charts

| Field | Value |
|---|---|
| Method | GET |
| URL | `/dashboard/charts` |
| Parameters | None |

**Expected Response:** 200 — chart data including severity distribution, disaster type breakdown, priority distribution, confidence histogram

---

### GET /dashboard/timeline

| Field | Value |
|---|---|
| Method | GET |
| URL | `/dashboard/timeline` |
| Parameters | `limit`: int 1–100 (default 20) |

**Expected Response:** 200

```json
[
  {
    "event_id": "evt-001",
    "title": "Hurricane Harvey",
    "type": "flood",
    "severity": 0.7,
    "timestamp": "2026-07-09T12:00:00Z",
    "analyzed": true
  }
]
```

---

### GET /dashboard/recent

| Field | Value |
|---|---|
| Method | GET |
| URL | `/dashboard/recent` |
| Parameters | `limit`: int 1–50 (default 10) |

**Expected Response:** 200

```json
[
  {
    "record_id": "uuid-string",
    "record_type": "batch",
    "severity": 72.3,
    "confidence": 85.1,
    "priority": "high",
    "disaster_type": "flood",
    "source": "batch_processing",
    "timestamp": "2026-07-09T10:30:00Z"
  }
]
```

---

## Reports

### GET /report/manual/{analysis_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/report/manual/{analysis_id}` |
| Parameters | `analysis_id`: path (required) |

Generate and download a PDF report for a manual (single-image) analysis.

**Expected Response:** 200 — `application/pdf` binary download

```
Content-Disposition: attachment; filename=crisismind_manual_{analysis_id}.pdf
```

---

### GET /report/live/{analysis_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/report/live/{analysis_id}` |
| Parameters | `analysis_id`: path (required) |

Generate and download a PDF report for a live disaster analysis.

**Expected Response:** 200 — `application/pdf` binary download

```
Content-Disposition: attachment; filename=crisismind_live_{analysis_id}.pdf
```

---

### GET /report/batch/{batch_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/report/batch/{batch_id}` |
| Parameters | `batch_id`: path (required) |

Generate and download a PDF report for an entire batch job.

**Expected Response:** 200 — `application/pdf` binary download

```
Content-Disposition: attachment; filename=crisismind_batch_{batch_id}.pdf
```

---

### GET /report/csv/{batch_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/report/csv/{batch_id}` |
| Parameters | `batch_id`: path (required) |

Generate and download a CSV of batch results.

**Expected Response:** 200 — `text/csv` download

```
Content-Disposition: attachment; filename=crisismind_batch_{batch_id}.csv
```

---

## Explainability

### POST /explainability/gradcam

| Field | Value |
|---|---|
| Method | POST |
| URL | `/explainability/gradcam` |
| Content-Type | `multipart/form-data` |
| Parameters | `image`: file (required), `tweet_text`: string (optional), `analysis_id`: string (optional) |

**Input:**
- `image`: JPEG/PNG/WebP/BMP from `testing/manual_test/` or `testing/sample_images/`
- `tweet_text`: (optional) — if provided, uses multimodal pipeline
- `analysis_id`: (optional) — associate explanation with a prior analysis

**Expected Response:** 200

```json
{
  "analysis_id": "uuid-string",
  "original": "data:image/png;base64,...",
  "heatmap": "data:image/png;base64,...",
  "overlay": "data:image/png;base64,..."
}
```

**Error responses:**
- 413 — image exceeds 10 MB
- 422 — unsupported image type or invalid file
- 503 — engine not ready

---

### GET /explainability/outputs/{analysis_id}/{filename}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/explainability/outputs/{analysis_id}/{filename}` |
| Parameters | `analysis_id`: path (required), `filename`: path (required) |

Serve a previously saved explanation image (PNG/JPG).

**Expected Response:** 200 — image binary (`image/png` or `image/jpeg`)

**Error responses:**
- 404 — file not found

---

## Analysis Center

### GET /analysis-center

| Field | Value |
|---|---|
| Method | GET |
| URL | `/analysis-center` |
| Parameters | `page`: int (default 1), `per_page`: int (default 20), `sort_by`: string (`newest`/`oldest`/`severity`/`confidence`), `query`: string (optional), `record_type`: string (optional: `manual`/`batch`/`live`), `analysis_mode`: string (optional), `severity`: string (optional: `little`/`mild`/`severe`/`critical`), `priority`: string (optional: `low`/`medium`/`high`/`critical`), `disaster_type`: string (optional), `source`: string (optional), `date_from`: string (optional, ISO date), `date_to`: string (optional, ISO date) |

**Expected Response:** 200

```json
{
  "records": [
    {
      "analysis_id": "uuid",
      "type": "manual",
      "severity": 65.0,
      "confidence": 88.2,
      "priority": "medium",
      "disaster_type": "flood",
      "source": "manual",
      "analysis_mode": "MULTIMODAL",
      "country": "USA",
      "timestamp": "2026-07-09T10:00:00Z",
      "has_explanation": true
    }
  ],
  "total": 42,
  "page": 1,
  "per_page": 10,
  "total_pages": 5
}
```

---

### GET /analysis-center/search

| Field | Value |
|---|---|
| Method | GET |
| URL | `/analysis-center/search` |
| Parameters | `q`: string (required, min 1 char), `page`: int (default 1), `per_page`: int (default 20), `sort_by`: string (default `newest`) |

**Expected Response:** 200

```json
{
  "records": [
    { "analysis_id": "uuid", "severity": 45.0, "disaster_type": "flood", ... }
  ],
  "total": 20
}
```

---

### GET /analysis-center/export

| Field | Value |
|---|---|
| Method | GET |
| URL | `/analysis-center/export` |
| Parameters | `query`, `record_type`, `analysis_mode`, `severity`, `priority`, `disaster_type`, `source`, `date_from`, `date_to` (all optional — same filters as `/analysis-center`) |

**Expected Response:** 200 — `text/csv` download

---

### GET /analysis-center/statistics

| Field | Value |
|---|---|
| Method | GET |
| URL | `/analysis-center/statistics` |
| Parameters | None |

**Expected Response:** 200

```json
{
  "total_analyses": 42,
  "by_type": { "manual": 10, "batch": 30, "live": 2 },
  "by_severity": { "little": 15, "mild": 12, "severe": 15 },
  "by_priority": { "low": 10, "medium": 15, "high": 12, "critical": 5 },
  "by_disaster_type": { "flood": 20, "earthquake": 10, "wildfire": 12 }
}
```

---

### GET /analysis-center/{analysis_id}

| Field | Value |
|---|---|
| Method | GET |
| URL | `/analysis-center/{analysis_id}` |
| Parameters | `analysis_id`: path (required) |

**Expected Response:** 200 — full record detail including `raw_data`, `recommended_actions`, `infrastructure_impact`, `emergency_level`, `has_explanation`

---

## Quick Reference — All 27 Endpoints

| # | Method | Path | Source File |
|---|--------|------|-------------|
| 1 | GET | `/health` | `main.py` |
| 2 | POST | `/predict` | `main.py` |
| 3 | GET | `/batch/status/{batch_id}` | `batch_routes.py` |
| 4 | GET | `/batch/results/{batch_id}` | `batch_routes.py` |
| 5 | GET | `/batch/download/csv/{batch_id}` | `batch_routes.py` |
| 6 | GET | `/batch/download/pdf/{batch_id}` | `batch_routes.py` |
| 7 | POST | `/batch/upload` | `batch_routes.py` |
| 8 | POST | `/batch/start/{batch_id}` | `batch_routes.py` |
| 9 | GET | `/disasters/events` | `disaster_routes.py` |
| 10 | GET | `/disasters/events/{event_id}` | `disaster_routes.py` |
| 11 | POST | `/disasters/events/{event_id}/analyze` | `disaster_routes.py` |
| 12 | GET | `/dashboard/summary` | `dashboard_routes.py` |
| 13 | GET | `/dashboard/statistics` | `dashboard_routes.py` |
| 14 | GET | `/dashboard/charts` | `dashboard_routes.py` |
| 15 | GET | `/dashboard/timeline` | `dashboard_routes.py` |
| 16 | GET | `/dashboard/recent` | `dashboard_routes.py` |
| 17 | GET | `/report/manual/{analysis_id}` | `report_routes.py` |
| 18 | GET | `/report/live/{analysis_id}` | `report_routes.py` |
| 19 | GET | `/report/batch/{batch_id}` | `report_routes.py` |
| 20 | GET | `/report/csv/{batch_id}` | `report_routes.py` |
| 21 | POST | `/explainability/gradcam` | `explain_routes.py` |
| 22 | GET | `/explainability/outputs/{analysis_id}/{filename}` | `explain_routes.py` |
| 23 | GET | `/analysis-center` | `analysis_center_routes.py` |
| 24 | GET | `/analysis-center/search` | `analysis_center_routes.py` |
| 25 | GET | `/analysis-center/export` | `analysis_center_routes.py` |
| 26 | GET | `/analysis-center/statistics` | `analysis_center_routes.py` |
| 27 | GET | `/analysis-center/{analysis_id}` | `analysis_center_routes.py` |
