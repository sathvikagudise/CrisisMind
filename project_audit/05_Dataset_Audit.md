# 05 — Dataset Audit

## Dataset: CrisisMMD v2.0

### Location
`dataset_analysis/CrisisMMD_v2.0/`

### Composition

| Metric | Value |
|---|---|
| Total image-text pairs | 18,082 |
| Human-annotated severity labels | 3,526 |
| Disaster events | 7 |
| Annotation TSV files | 7 |
| Image directories | 7 |

### Event Breakdown

| Event | TSV | Images | Annotated |
|---|---|---|---|
| California Wildfires | `california_wildfires.tsv` | ✅ | ✅ |
| Hurricane Harvey | `hurricane_harvey.tsv` | ✅ | ✅ |
| Hurricane Irma | `hurricane_irma.tsv` | ✅ | ✅ |
| Hurricane Maria | `hurricane_maria.tsv` | ✅ | ✅ |
| Iraq Iraq Iran Earthquake | `iraq_iran_earthquake.tsv` | ✅ | ✅ |
| Mexico Earthquake | `mexico_earthquake.tsv` | ✅ | ✅ |
| Sri Lanka Floods | `sri_lanka_floods.tsv` | ✅ | ✅ |

### TSV Files

All 7 TSVs present, totalling ~6.6 MB. Each TSV contains columns for `image_id`, `tweet_id`, `image_path`, `tweet_text`, `severity`, `humanitarian`, `informative`, plus additional metadata.

### Label Distribution (Severity-Annotated Subset)

| Label | Count | Percentage |
|---|---|---|
| Little or No Damage | 475 | 13.47% |
| Mild Damage | 838 | 23.77% |
| Severe Damage | 2,211 | 62.74% |
| (dont_know excluded) | 2 | — |

**Imbalance**: Severe Damage is ~62.7% of annotated samples. This 4.7:1 ratio is addressed by class weighting in all training scripts.

### Image Files

- All image paths verified by `dataset_audit.py`: **18,082/18,082 (100%) exist**
- Total image storage: not measured (excluded from audit scope)
- Formats: JPEG/PNG mixed
- Missing images: **0**

### Verification Script

`dataset_audit.py` (266 lines) performs:
- TSV loading with Mac dotfile exclusion
- Per-file row/column/missing statistics
- Label distribution across 5 target columns (severity, humanitarian, informative, image_human, text_human)
- Image path existence verification
- 3 distribution plots (severity, humanitarian categories, event-wise)
- Markdown report generation

### Scripts Using the Dataset

| Script | TSV Source | Uses Cached Features? |
|---|---|---|
| `run_image_baseline.py` | Direct load | No |
| `run_text_baseline.py` | Direct load | No |
| `run_fusion.py` | Direct load | Yes (pre-extracted) |
| `datasets/prepare_data.py` | Direct load | No |
| `datasets/multimodal_data.py` | Direct load | No |

### Issues

| # | Severity | Issue |
|---|---|---|
| 1 | **LOW** | 2 samples with `dont_know` severity label exist; handled inconsistently (excluded in some scripts, counted in others) |
| 2 | **LOW** | `dataset_audit.py` uses Mac dot-file filter (`._*`) — not portable to Linux |
| 3 | **LOW** | Severity percentage discrepancy between research docs (62.74% excludes `dont_know`) and dataset summary report (60.99% includes it) |
| 4 | **LOW** | No dataset version checksum or hash — silent data corruption not detectable |
| 5 | **LOW** | 4 duplicate rows discovered in `sri_lanka_floods.tsv` |

## Overall Dataset Score: **9.0 / 10**
