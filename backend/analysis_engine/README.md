# Analysis Engine

One centralized entry point for all AI inference across CrisisMind.

## Architecture

```
                   ┌──────────────────────────────┐
                   │      AnalysisRequest          │
                   │  image | text | metadata      │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │        Validators             │
                   │  check inputs → determine     │
                   │  analysis mode                │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │      Model Selector           │
                   │  image+text → GatedFusion     │
                   │  image      → EfficientNet    │
                   │  text       → DistilBERT      │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │      Inference                │
                   │  run selected model           │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │      Decision Engine          │
                   │  severity → priority          │
                   │          → actions            │
                   │          → emergency level    │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │      Response Builder         │
                   │  standardised JSON response   │
                   └──────────────────────────────┘
```

## Request Flow

1. **AnalysisRequest** — wraps image (PIL), text, metadata, source.
2. **Validators** — ensure inputs are present for the detected or requested mode.
3. **Model Selector** — automatically picks the right model based on input modality.
4. **Inference** — runs the selected model (`GatedFusion`, `EfficientNet`, or `DistilBERT`).
5. **Decision Engine** — rule-based mapping from predicted severity to response actions.
6. **Response Builder** — assembles the final unified JSON.

## Modes

| Mode         | Inputs       | Model             |
|--------------|------------|-------------------|
| IMAGE_ONLY   | image       | EfficientNet-B0   |
| TEXT_ONLY    | text        | DistilBERT        |
| MULTIMODAL   | image+text  | GatedFusion v1    |

## Decision Rules

| Severity              | Priority | Emergency Level          | Infrastructure |
|----------------------|----------|--------------------------|----------------|
| Little or No Damage  | Low      | Monitor Situation        | Minor          |
| Mild Damage          | Medium   | Field Assessment Required | Moderate       |
| Severe Damage        | High     | Immediate Response       | Major          |

## API Integration

```python
from backend.analysis_engine import AnalysisEngine, AnalysisRequest

engine = AnalysisEngine(model_dict)
result = engine.analyze(AnalysisRequest(image=..., text=...))
```

The existing `POST /predict` endpoint now internally delegates to `engine.analyze()`.

## Disaster Event Hook

```python
from backend.analysis_engine.disaster_hook import analyze_disaster_event

result = analyze_disaster_event(disaster_event, engine)
```

Converts a `DisasterEvent` into an `AnalysisRequest` and runs it through the engine. The provider pipeline is not yet connected.

## Future Integrations

- **GDACS provider** — wire disaster events automatically through the analysis hook.
- **Redis cache** — cache analysis results for repeated events.
- **Batch processing** — extend `AnalysisEngine` to accept lists of requests.
- **Frontend components** — disaster feed with inline severity analysis.
