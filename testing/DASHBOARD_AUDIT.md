# Dashboard Audit — CrisisMind v1.0

**Date:** 2026-07-23
**Scope:** Dashboard, Analysis Center, Disasters pages + backend data pipeline
**Status:** All issues found and fixed

---

## Issues Found & Fixes Applied

### 1. HTML Entity Decoding

**Problem:** ReliefWeb API returns HTML-encoded strings (`&#8217;`, `&amp;`, `&lt;b&gt;`, etc.) in descriptions and titles. NASA EONET titles like `DEVIL&#039;S` were displayed with raw entity references.

**Root cause:** ReliefWeb `_safe_html_to_text()` strips HTML tags but does not decode HTML character references. NASA provider passes titles through without sanitization.

**Fix (3 files):**
- `backend/disaster_intelligence/providers/reliefweb_provider.py:224-231` — Added `html.unescape()` after tag stripping in `_safe_html_to_text()`
- `backend/disaster_intelligence/providers/nasa_provider.py:106-151` — Added `html_mod.unescape()` to title, description, and country in `normalize()`
- `backend/disaster_intelligence/normalizer.py:36-50` — Added `html_mod.unescape()` as cross-cutting decode on title, description, and country in the normalizer (catches all providers)

### 2. AI Models Loaded Count

**Problem:** Hardcoded `"5 models loaded"` string in Dashboard System Health section. Not derived from actual backend state.

**Fix (3 files):**
- `backend/api/schemas.py:23-24` — Extended `HealthResponse` with `models_loaded: int = 0`
- `backend/api/main.py:80-82` — Store `len(models)` in `app.state.models_loaded` at startup
- `backend/api/main.py:91-93` — Return `models_loaded` from `/health` endpoint
- `frontend/src/types/index.ts:14-17` — Added `models_loaded: number` to `HealthResponse` type
- `frontend/src/pages/Dashboard/index.tsx:524` — Read `health.models_loaded` dynamically instead of hardcoded value

### 3. Confidence Double-Scaling Bug

**Problem:** The analysis engine returns confidence as 0-100 (e.g. 65.43 = 65.43%). AnalysisCenter table rows and detail modal multiplied by 100 again, producing impossible values like 6543%.

**Root cause:** `AnalysisCenter/index.tsx` lines 387 and 528 used `(record.confidence * 100)` assuming 0-1 fraction, but confidence is already 0-100 from `engine.py:130`.

**Fix:**
- `frontend/src/pages/AnalysisCenter/index.tsx:387` — Changed `(record.confidence * 100).toFixed(0)` to `record.confidence.toFixed(1)`
- `frontend/src/pages/AnalysisCenter/index.tsx:528` — Changed `(detail.confidence * 100).toFixed(1)` to `detail.confidence.toFixed(1)`

### 4. Confidence Formatting Standardization

**Problem:** Inconsistent precision across pages: Dashboard used `.toFixed(1)`, AnalysisCenter table used `.toFixed(0)`, AnalysisCenter detail used `.toFixed(1)`.

**Fix:** All confidence values now use `.toFixed(1)%` format (e.g. "53.8%") across:
- Dashboard StatsSection (`Dashboard/index.tsx:207`)
- AnalysisCenter table rows
- AnalysisCenter detail modal
- AnalysisCenter statistics bar
- Disasters EmergencyAssessmentCard (`Disasters/index.tsx:415`)

### 5. UUID Truncation

**Problem:** Raw UUIDs (36 chars) displayed as fallback text in Dashboard Activity Timeline and Analysis Center table rows, consuming excessive horizontal space.

**Fix:**
- `frontend/src/pages/Dashboard/index.tsx` — Added `truncateId()` helper (first 8 chars + ellipsis); applied to timeline entry fallback
- `frontend/src/pages/AnalysisCenter/index.tsx` — Added `truncateId()` helper; applied to table row title fallback
- Detail modal ID field left full-length (appropriate for detail view)

### 6. Timestamp Formatting

**Problem:** Dashboard timeline used time-only format ("2:30 PM"), AnalysisCenter used date+time ("Jul 15, 2:30 PM"). Inconsistent for same data type.

**Fix:**
- `frontend/src/pages/Dashboard/index.tsx:358-363` — Changed `formatTime()` from `toLocaleTimeString` (time only) to `toLocaleDateString` with `{ month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }` format, matching AnalysisCenter

### 7. "Unknown" Location Labels

**Problem:** Backend validator defaulted empty country to `"Unknown"` string. Frontend displayed "Unknown" for missing locations — generic and misleading.

**Fix (3 files):**
- `backend/disaster_intelligence/validators.py:62-66` — Changed default from `"Unknown"` to `""` (empty string)
- `frontend/src/pages/Dashboard/index.tsx:462` — `{evt.country || "Location unavailable"}`
- `frontend/src/pages/Disasters/index.tsx:308` — `{event.country || "Location unavailable"}`
- `frontend/src/pages/Disasters/index.tsx:429` — `{evt.country || "Location unavailable"}`
- `frontend/src/pages/Disasters/index.tsx:76` — Changed date fallback from `"Unknown"` to `"Date unavailable"`

### 8. Dashboard Statistics Consistency

**Analysis:** Dashboard statistics are computed from `DashboardStore.get_all()` (tracks manual, batch, live analyses as they occur). AnalysisCenter statistics merge `DashboardStore` + `AnalysisStore`. These are intentionally different data sources — Dashboard tracks explicitly, AnalysisCenter is a unified view.

**Verification:** All three tracking points confirmed correct:
- Manual: `main.py:181` → `dashboard.track_manual(record)` ✅
- Batch: `batch_routes.py:52` → `dashboard.track_batch(record)` ✅
- Live: `disaster_routes.py:245` → `dashboard.track_live(record)` ✅

No changes needed — data flow is correct.

### 9. Disaster Types Chart

**Analysis:** Chart data from `analytics.py` uses `Counter(r.disaster_type)` from stored records. Only shows types that have been analyzed (correct behavior for an analysis dashboard). Disaster type strings come from NASA `CATEGORY_MAP` which standardizes names (e.g. "Wildfire", "Flood", "Earthquake").

**Verification:** Chart correctly represents all disaster categories among analyzed records. No changes needed.

### 10. Charts/Cards/Stats Update After Analysis

**Analysis:** Dashboard refreshes data via `loadAll()` on page load and manual Refresh button. After performing analyses (manual, batch, live), the dashboard store is updated immediately via tracking calls. The next page load/refresh picks up all new data.

**Verification:** Data flow confirmed:
- Manual analysis → tracks in `main.py` → dashboard shows on refresh ✅
- Batch analysis → tracks in `batch_routes.py` → dashboard shows on refresh ✅
- Live analysis → tracks in `disaster_routes.py` → dashboard shows on refresh ✅

No auto-refresh needed (by design — research prototype).

---

## Files Modified

| File | Changes |
|------|---------|
| `backend/api/schemas.py` | Added `models_loaded` to HealthResponse |
| `backend/api/main.py` | Store model count at startup; return in /health |
| `backend/disaster_intelligence/normalizer.py` | Added `html.unescape()` for title/description/country |
| `backend/disaster_intelligence/providers/reliefweb_provider.py` | Added `html.unescape()` in `_safe_html_to_text()` |
| `backend/disaster_intelligence/providers/nasa_provider.py` | Added `html.unescape()` for title/description/country |
| `backend/disaster_intelligence/validators.py` | Changed country default from "Unknown" to "" |
| `frontend/src/types/index.ts` | Added `models_loaded` to HealthResponse |
| `frontend/src/pages/Dashboard/index.tsx` | Dynamic model count, UUID truncation, timestamp format, location fallback |
| `frontend/src/pages/AnalysisCenter/index.tsx` | Fixed confidence double-scaling, UUID truncation, standardized .toFixed(1)% |
| `frontend/src/pages/Disasters/index.tsx` | Location fallback, date fallback |

---

## Validation

- **TypeScript:** Zero errors (`npx tsc --noEmit`)
- **Python imports:** All backend modules import successfully
- **Confidence values:** All display locations use 0-100 scale with `.toFixed(1)%` format
- **HTML entities:** Decoded server-side before storage (normalizer + providers)
- **UUIDs:** Truncated to 8 chars + ellipsis in list views, full in detail modal
- **Timestamps:** All use "MMM D, HH:MM AM/PM" format consistently
- **Location labels:** "Location unavailable" for missing countries, "Date unavailable" for missing dates
