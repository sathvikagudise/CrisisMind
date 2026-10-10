# Disaster Intelligence Module

A pluggable, provider-agnostic module for fetching disaster events from
public APIs and normalising them into a unified schema for downstream
consumption by the CrisisMind AI pipeline.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   DisasterIntelligenceService                     │
│                                                                   │
│  ┌───────────┐   ┌──────────────┐   ┌────────────┐   ┌────────┐ │
│  │ Provider   │──→│ normalize()  │──→│ Validator  │──→│ Cache  │ │
│  │ (ABC)      │   │ (per-source) │   │ (cross-cut)│   │ (30m)  │ │
│  └───────────┘   └──────────────┘   └────────────┘   └────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Folder Structure

```
backend/disaster_intelligence/
    __init__.py              # Package marker
    config.py                # Settings (env-driven)
    models.py                # Domain models (ProviderStatus)
    schemas.py               # DisasterEvent Pydantic schema
    cache.py                 # Cache ABC + MemoryCache implementation
    exceptions.py            # Custom exception hierarchy
    service.py               # Orchestration entry point
    normalizer.py            # Cross-provider normalisation
    validators.py            # EventValidator (required-field checks)
    providers/
        __init__.py
        base_provider.py     # Abstract base class
        reliefweb_provider.py  # ReliefWeb API v2 (requires appname)
        nasa_provider.py       # NASA EONET (free, no auth)
        gdacs_provider.py      # Skeleton only
    README.md
    architecture.md

backend/api/
    disaster_routes.py       # GET /disasters/events
                             # GET /disasters/events/{event_id}
```

## Responsibilities

| Layer | Responsibility |
|---|---|
| `config.py` | Load environment variables, expose typed settings |
| `schemas.py` | Define `DisasterEvent` — the single currency type |
| `base_provider.py` | Abstract interface every provider must implement |
| `providers/*.py` | API-specific fetch, normalize, validate logic |
| `validators.py` | Cross-cutting checks (title, country, URL, date) |
| `cache.py` | Cache ABC + MemoryCache (TTL-based eviction) |
| `service.py` | Load provider, fetch, normalise, validate, cache |
| `exceptions.py` | Typed errors for every failure mode |

## Data Flow

```
External API  ──→  Provider.fetch_events()
                         │
                         ▼
                    raw dict(s)
                         │
                         ▼
                   Provider.normalize(item)
                         │
                         ▼
                   DisasterEvent(s)
                         │
                         ▼
                   EventValidator.validate(event)
                         │
                         ▼
                   EventValidator fixes defaults / drops corrupted
                         │
                         ▼
                   MemoryCache.set(key, result, ttl=1800)
                         │
                         ▼
                   DisasterIntelligenceService
                         │
                         ▼
              FastAPI  |  AI Pipeline  |  Frontend
```

## ReliefWeb Provider

**Endpoint:** `https://api.reliefweb.int/v2/disasters`

**Status:** Implemented — requires a pre-approved appname.

Since November 2025 the ReliefWeb API v2 mandates a registered appname
for every request.  To use this provider:

1. Register at https://apidoc.reliefweb.int/parameters#appname
2. Set `CM_DI_RELIEFWEB_APPNAME=crisismind-<your-random-suffix>`

Until the appname is approved, the module falls back to NASA EONET:

```
set CM_DI_DEFAULT_PROVIDER=nasa
```

## NASA EONET Provider (default for testing)

**Endpoint:** `https://eonet.gsfc.nasa.gov/api/v3/events`

**Status:** Working — free, no authentication required.

Returns natural events (wildfires, storms, volcanoes, etc.) with
coordinates, dates, categories, and source URLs.  Country is extracted
from the event title or the optional `countries` field.

## Validator

`backend/disaster_intelligence/validators.py`

| Check | Rule |
|---|---|
| Title | Must be ≥ 3 characters |
| Description | Must be ≥ 5 characters |
| Country | Defaults to "Unknown" if missing |
| source_url | Must be a valid HTTP(S) URL |
| Date | Must be a `datetime` object if present |
| Coordinates | Out-of-bounds → reset to None |

## Caching

`MemoryCache` is active by default with a **30-minute TTL** (1800 s).

Cache events are logged:
- `Cache HIT provider=nasa key=nasa:events events=10`
- `Cache MISS provider=nasa key=nasa:events — calling API`

## FastAPI Routes

| Method | Path | Description | Params |
|---|---|---|---|
| GET | `/disasters/events` | List disaster events | `limit` (1-100, default 20), `disaster_type`, `country` |
| GET | `/disasters/events/{event_id}` | Single event detail | — |

### Sample Response (list)

```json
[
  {
    "id": "nasa-EONET_21003",
    "title": "Wildfire Maze, Madison, Idaho",
    "type": "Wildfires",
    "country": "Idaho",
    "date": "2026-07-06T17:58:00+00:00",
    "description": "6 Miles W from Rexburg, ID",
    "source": "nasa",
    "status": "NEW"
  }
]
```

### Sample Response (detail)

```json
{
  "id": "nasa-EONET_21003",
  "title": "Wildfire Maze, Madison, Idaho",
  "type": "Wildfires",
  "country": "Idaho",
  "date": "2026-07-06T17:58:00+00:00",
  "description": "6 Miles W from Rexburg, ID",
  "source": "nasa",
  "source_url": "https://irwin.doi.gov/observer/incidents/2026-IDIFD-000096",
  "latitude": 43.828883,
  "longitude": -111.956783,
  "image_url": null,
  "severity": null,
  "status": "NEW",
  "metadata": { "nasa_id": "EONET_21003", "categories": ["Wildfires"], "closed": null }
}
```

## Error Handling

| Scenario | HTTP Status | Detail |
|---|---|---|
| Provider unreachable | 503 | Provider name + reason |
| Rate limited | 429 | Retry guidance |
| No data | 404 | Empty result message |
| Invalid request | 422 | Validation details |

## Environment Variables

All variables use the `CM_DI_` prefix.

| Variable | Default | Description |
|---|---|---|
| `CM_DI_DEFAULT_PROVIDER` | `reliefweb` | Active provider name |
| `CM_DI_MAX_EVENTS_PER_REQUEST` | `50` | Max events per fetch |
| `CM_DI_REQUEST_TIMEOUT_SECONDS` | `30` | HTTP timeout |
| `CM_DI_RETRY_COUNT` | `3` | Retry attempts |
| `CM_DI_CACHE_DURATION_SECONDS` | `1800` | Cache TTL (30 min) |
| `CM_DI_CACHE_MAX_SIZE` | `1000` | Max cached entries |
| `CM_DI_RELIEFWEB_APPNAME` | — | ReliefWeb v2 appname |
| `CM_DI_RELIEFWEB_API_KEY` | — | ReliefWeb API key |
| `CM_DI_NASA_API_KEY` | — | NASA API key |
| `CM_DI_GDACS_API_KEY` | — | GDACS API key |

## Adding a New Provider

1.  Create `providers/my_provider.py`.
2.  Subclass `BaseProvider` and implement all abstract methods.
3.  Add an `elif` branch in `service.py:load_provider()`.
4.  Add the API key to `config.py`.
5.  No changes to the AI pipeline, schemas, or normalizer.

## Future Integrations

1.  **ReliefWeb appname registration** — restore as primary provider
2.  **GDACS provider** — implement HTTP wiring
3.  **Redis cache** — implement `Cache` ABC
4.  **AI pipeline** — feed `DisasterEvent` through GatedFusion
5.  **Frontend** — disaster feed, event cards, map view
