# Disaster Intelligence — Architecture Document

## Overview

The Disaster Intelligence module provides a clean abstraction over
disaster-event data sources.  Its primary goals are:

- **Decouple data acquisition from AI inference.**
- **Make every data source interchangeable** through a single abstract
  interface.
- **Normalise all data into one schema** so that downstream code never
  depends on a specific API.

---

## How Providers Connect

```
                    ┌──────────────────────┐
                    │   BaseProvider       │  ← Abstract interface
                    │   (ABC)             │
                    └──────┬───────────────┘
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
 ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
 │ ReliefWeb      │ │ NASA (EONET) │ │ GDACS        │
 │ Provider       │ │ Provider     │ │ Provider     │
 │                │ │              │ │              │
 │ fetch_events() │ │ fetch_events │ │ fetch_events │
 │ fetch_event()  │ │ fetch_event()│ │ fetch_event()│
 │ normalize()    │ │ normalize()  │ │ normalize()  │
 │ validate()     │ │ validate()   │ │ validate()   │
 └────────────────┘ └──────────────┘ └──────────────┘
```

Each provider inherits from `BaseProvider` and maps its own API response
fields to the `DisasterEvent` schema inside `normalize()`.

---

## How the Service Works

```
                    DisasterIntelligenceService
                              │
                    load_provider("reliefweb")
                              │
                    get_events(disaster_type="flood")
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Cache.get()    Provider        Cache.set()
                        fetch_events()
                              │
                              ▼
                        Normalizer
                        normalize()
                              │
                              ▼
                        Provider
                        validate()
                              │
                              ▼
                     DisasterEventList
```

The service is the sole entry point.  It:

1.  Checks the cache first.
2.  Delegates to the active provider's `fetch_events()`.
3.  Normalises each raw response via the **provider's own** `normalize()` (field-mapping).
4.  Validates every event via `EventValidator` (cross-cutting required-field checks).
5.  Stores the result in `MemoryCache` (30-minute TTL).
6.  Returns a `DisasterEventList`.

---

## How Normalisation Works

```
  ReliefWeb API            NASA EONET              GDACS
      │                       │                      │
      ▼                       ▼                      ▼
  {id: "...",             {id: "...",             {id: "...",
   headline: "...",        title: "...",            title: "...",
   description: "...",     description: "...",      description: "...",
   ...}                    ...}                     ...}
      │                       │                      │
      └───────────────┬───────┴──────────┬───────────┘
                      ▼                  ▼
               ReliefWebProvider    NASAProvider
               .normalize()         .normalize()
                      │                  │
                      └────────┬─────────┘
                               ▼
                     EventNormalizer
                     .normalize()
                               │
                               ▼
                      DisasterEvent
                      {event_id, title, description,
                       disaster_type, country, ...}
```

Each provider maps its API's fields to the common schema inside its own
`normalize()`.  The `EventValidator` then performs cross-cutting
required-field checks on the resulting `DisasterEvent` objects.

---

## Class Diagram

```
┌─────────────────────────────┐
│     DisasterIntelligence    │
│     Service                 │
├─────────────────────────────┤
│ - cache: Cache              │
│ - normalizer: Normalizer    │
│ - provider: BaseProvider    │
├─────────────────────────────┤
│ + load_provider(name)       │
│ + get_events()              │
│ + get_event()               │
│ + provider (property)       │
└──────────┬──────────────────┘
           │
           │ uses ───────────────┐
           │                     │
           ▼                     ▼
┌────────────────────┐  ┌───────────────┐
│   BaseProvider     │  │  EventNormal- │
│   (ABC)            │  │  izer         │
├────────────────────┤  ├───────────────┤
│ + provider_name    │  │ + normalize() │
│ + fetch_events()   │  └───────────────┘
│ + fetch_event()    │
│ + normalize()      │        ▲
│ + validate()       │        │
└────────┬───────────┘        │
         │                    │
         │ implements         │
    ┌────┴─────┬──────┐       │
    ▼          ▼      ▼       │
┌────────┐┌────────┐┌────┐   │
│Relief- ││ NASA   ││G-  │   │
│Web     ││        ││DACS│   │
└────────┘└────────┘└────┘   │
                              │
┌──────────────────┐          │
│    Cache (ABC)   │          │
├──────────────────┤          │
│ + get()          │          │
│ + set()          │          │
│ + invalidate()   │          │
│ + clear()        │          │
└──────────────────┘          │
                              │
┌──────────────────┐          │
│ DisasterEvent    │◄─────────┘
│ (Pydantic)       │
├──────────────────┤
│ + event_id       │
│ + title          │
│ + description    │
│ + disaster_type  │
│ + country        │
│ + latitude       │
│ + longitude      │
│ + event_date     │
│ + image_url      │
│ + source         │
│ + source_url     │
│ + severity       │
│ + metadata       │
└──────────────────┘

┌─────────────────────────┐
│ DisasterIntelligence    │
│ Config (BaseSettings)   │
├─────────────────────────┤
│ default_provider        │
│ max_events              │
│ timeout / retry / cache │
│ api_keys                │
└─────────────────────────┘

┌─────────────────────────┐
│ Exceptions Hierarchy    │
├─────────────────────────┤
│ DisasterIntelligenceError│
│  ├ ProviderUnavailable  │
│  ├ InvalidResponse      │
│  ├ RateLimitExceeded    │
│  ├ InvalidConfiguration │
│  └ NoDataAvailable      │
└─────────────────────────┘
```

---

## Folder Diagram

```
backend/disaster_intelligence/
├── __init__.py                # Package init
├── README.md                  # Module documentation
├── architecture.md            # This file
├── config.py                  # CM_DI_* env settings
├── models.py                  # ProviderHealth, ProviderStatus
├── schemas.py                 # DisasterEvent, DisasterEventList
├── cache.py                   # Cache ABC (get/set/invalidate/clear)
├── exceptions.py              # Custom exception classes
├── service.py                 # DisasterIntelligenceService
├── normalizer.py              # EventNormalizer (utility)
├── validators.py              # EventValidator (required-field checks)
└── providers/
    ├── __init__.py
    ├── base_provider.py       # BaseProvider ABC
    ├── reliefweb_provider.py  # ReliefWeb v2 (requires appname)
    ├── nasa_provider.py       # NASA EONET (working, no auth)
    └── gdacs_provider.py      # GDACS skeleton
```

---

## Future API Integration

Both `ReliefWebProvider` and `NASAProvider` are **fully implemented**:

- They use `httpx` for HTTP calls with retry/timeout logic.
- They parse the JSON response and call `normalize()` on each item.
- They wrap network errors in the custom exception hierarchy.
- ReliefWeb requires a registered appname (register via Google Form).
- NASA EONET works without authentication and is the default for testing.

---

## Key Architecture Decisions

| Decision | Rationale |
|---|---|
| **Provider as Strategy pattern** | Swap data sources without touching AI code |
| **Pydantic DisasterEvent** | Runtime validation, clear typing, FastAPI compatible |
| **ABC for cache** | Memory → Redis → DB swap without code change |
| **Service as Facade** | Single entry point shields consumers from complexity |
| **Env-based config** | Twelve-Factor App compliance; no hardcoded secrets |
| **Lazy provider loading** | Only import the provider that's actually used |

---

## Extension Points

1.  **New provider** — subclass `BaseProvider`, register in service
2.  **New cache backend** — implement `Cache` ABC
3.  **New event schema fields** — extend `DisasterEvent` (Pydantic)
4.  **Pre/post processing hooks** — add to `DisasterIntelligenceService`
5.  **Batch operations** — extend `get_events()` with pagination
6.  **Event subscriptions** — add webhook/callback support

---

## Future Implementation Order

1.  ✅ **ReliefWeb provider** — implemented (awaiting appname approval)
2.  ✅ **NASA EONET provider** — implemented and working
3.  ✅ **MemoryCache** — implemented and active (30-min TTL)
4.  ✅ **FastAPI routes** — `GET /disasters/events` and `GET /disasters/events/{id}`
5.  ⬜ **GDACS provider** — HTTP wiring
6.  ⬜ **Redis cache** — implement `Cache` ABC
7.  ⬜ **AI pipeline integration** — feed `DisasterEvent` into GatedFusion
8.  ⬜ **Frontend** — disaster feed, event cards, map view
