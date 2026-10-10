# Cleaning Summary — CrisisMMD v2.0 Severity Subset

**Phase 3 deliverable.** Applies the pre-registered, objective cleaning rules defined in
[`DATASET_AUDIT_REPORT.md`](DATASET_AUDIT_REPORT.md) §4. All rules are deterministic and
independent of any model prediction.

## Scope

| Population | Samples |
|---|---|
| Original pipeline population (`load_multimodal_data()`) | 3,526 |
| **Removed** | **421 (11.9%)** |
| **Kept (cleaned population)** | **3,105** |

## Removed samples by rule

| Rule | Category | Description | Removed |
|---|---|---|---|
| F | Exact-duplicate image | dHash distance 0.0 → keep highest-confidence member | 83 |
| G | Conflicting near-dup labels | Cluster (≤0.10) with conflicting severity labels → majority wins | 25 |
| H | Ambiguous label | `image_damage_conf < 0.6` | 313 |
| A/B/C/D/E | File/text quality | missing/corrupt/blank/tiny image, empty text, non-informative text | 0 |
| | **Total** | | **421** |

## Impact by class

| Class | Original | Kept | Removed |
|---|---|---|---|
| severe_damage | 2,212 | 2,013 | 199 |
| mild_damage | 839 | 689 | 150 |
| little_or_no_damage | 475 | 403 | 72 |
| **Total** | **3,526** | **3,105** | **421** |

Removed-by-rule × class (row: rule, cols: little/mild/severe):

| Rule | little | mild | severe |
|---|---|---|---|
| F | 3 | 19 | 61 |
| G | 9 | 10 | 6 |
| H | 60 | 121 | 132 |

## Confidence improvement

| Statistic | Original | Kept |
|---|---|---|
| Mean `image_damage_conf` | 0.793 | 0.837 |
| Min `image_damage_conf` | 0.275 | 0.602 |

## Artifacts produced

| Artifact | Location | Content |
|---|---|---|
| Original dataset snapshot | `dataset_analysis/CrisisMMD_v2.0_original/` | Full CrisisMMD v2.0 copy (annotations + json + Readme + `data_image/`) |
| Cleaned dataset | `dataset_analysis/CrisisMMD_v2.0_clean/` | Only kept images (3,105) + per-event severity-filtered TSVs + `MANIFEST.json` |
| Removed-samples ledger | `research/data_cleaning/REMOVED_SAMPLES.csv` | 421 rows: `sample_id`, `image_path`, `text`, `original_label`, `image_damage_conf`, `removal_category`, `removal_reason` |
| Kept-samples ledger | `research/data_cleaning/KEPT_SAMPLES.csv` | 3,105 rows: path, text, label, conf, (placeholder) split |
| Cleaning summary | `research/data_cleaning/CLEANING_SUMMARY.md` | This file |

> Note: `KEPT_SAMPLES.csv` is written before Phase 4; its `split` column is finalized in
> Phase 4 and the CSV is regenerated there with the leak-free split assignment.

## Verification notes

- All 3,526 source images exist and decode; no file was removed for read failure.
- Cleaned per-event TSVs were re-derived from the source TSVs by `image_path` membership
  and re-verified row counts against `KEPT_SAMPLES.csv` (e.g. california_wildfires: 467).
- No image file was modified, re-encoded, or regenerated — cleaning is selection-only.
