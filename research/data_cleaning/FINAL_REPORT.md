# CrisisMind — Data Quality Pipeline: Final Report

**Phase 9 deliverable.** Summary of the complete data-quality improvement pipeline
(audit → cleaning → split → retrain → evaluate → compare → diagram → integrity check)
for the CrisisMMD v2.0 severity-classification subset used by the Gated Fusion v3 model.

**Date:** 2026-08-14 · **Status:** Complete · **Integrity checks:** 29/29 passed

---

## 1. What was done (9 phases)

| Phase | Deliverable | Result |
|---|---|---|
| 1 · Audit | `DATASET_AUDIT_REPORT.md` | Full audit of 3,526-sample severity subset; root causes identified |
| 2 · Cleaning criteria | defined in audit report §4 | 7 objective rules (A–H), pre-registered, no model-prediction involvement |
| 3 · Cleaned dataset | `CrisisMMD_v2.0_original/`, `CrisisMMD_v2.0_clean/`, `REMOVED_SAMPLES.csv`, `CLEANING_SUMMARY.md` | 421 samples removed (11.9%); full traceability |
| 4 · Splits | `CLEANED_DATASET_SPLIT.md` + split files | Leak-free stratified 70/15/15 → 2,173/467/465 |
| 5 · Retrain | `step1_extract_clean.py`, `train_v3_clean.py`, model artifacts | Gated Fusion v3 retrained with the *identical* `wide_wce_g1.0_lr1e-04` recipe |
| 6 · Evaluate | `CLEANED_MODEL_REPORT.md`, `cleaned_dataset_results.csv` | Full metrics + latency |
| 7 · Compare | `ORIGINAL_VS_CLEANED_COMPARISON.md` | Original vs cleaned, honest interpretation |
| 8 · Diagram | `Architecture@crisismind.png` + `06_Project_Architecture.md` | Dataset/input portion updated to quality pipeline |
| 9 · Integrity | this report + 29-check verification | All checks passed |

## 2. Key numbers

| Metric | Original | Cleaned |
|---|---|---|
| Population | 3,526 | 3,105 (−421, 11.9%) |
| Duplicate image groups removed (F) | — | 83 |
| Conflicting near-dup labels removed (G) | — | 25 |
| Low-confidence removed (H, conf<0.6) | — | 313 |
| Near-dup clusters crossing splits | **94** | **0** |
| Exact-identical pairs crossing splits | **27** | **0** |
| Train / Val / Test | 2,468 / 529 / 529 | 2,173 / 467 / 465 |

### Model (same architecture, same recipe)

| Metric | Original v3 | Cleaned v3 | Δ |
|---|---|---|---|
| Accuracy | 0.6011 | 0.5699 | −0.031 |
| Macro F1 | 0.5372 | 0.4829 | −0.054 |
| Weighted F1 | 0.6220 | 0.5972 | −0.025 |
| Image Gate | 0.5365 | 0.5579 | +0.021 |
| Text Gate | 0.4357 | 0.4177 | −0.018 |

## 3. Interpretation

- The cleaned model scores **lower** because the original split leaked 94 near-duplicate
  clusters across the train/test boundary — its published metrics were optimistically
  biased. The cleaned test set (0 near-dup crossings, every cluster contained) is the
  honest generalization estimate.
- The architecture, encoders, loss, optimizer, scheduler, sampler, seed, and even
  early-stopping behavior (best epoch 6 in both runs) are unchanged; the delta is purely a
  dataset-integrity effect.
- Constraints honored: no architecture/encoder/decision-engine changes, no
  prediction-based removal, no metric-driven sample selection, all cleaning rules
  objective & reproducible, and the IEEE paper was **not** modified.

## 4. Reproducibility

Everything is re-runnable from source:

```bash
# 1–4: audit + cleaning + split (deterministic, seed 42)
# 5:    extract features
backend/.venv/Scripts/python.exe backend/training/step1_extract_clean.py
# 5:    retrain (identical recipe to wide_wce_g1.0_lr1e-04)
backend/.venv/Scripts/python.exe backend/training/train_v3_clean.py
# 6:    evaluate + latency
backend/.venv/Scripts/python.exe backend/training/eval_v3_clean.py
# 8:    regenerate architecture diagram
backend/.venv/Scripts/python.exe research/assets/architecture/generate_architecture_diagram.py
```

## 5. Artifacts index

| Artifact | Location |
|---|---|
| Audit report | `research/data_cleaning/DATASET_AUDIT_REPORT.md` |
| Cleaning summary | `research/data_cleaning/CLEANING_SUMMARY.md` |
| Removed-samples ledger | `research/data_cleaning/REMOVED_SAMPLES.csv` |
| Kept-samples ledger (with split) | `research/data_cleaning/KEPT_SAMPLES.csv` |
| Split documentation | `research/data_cleaning/CLEANED_DATASET_SPLIT.md` |
| Cleaned model report | `research/data_cleaning/CLEANED_MODEL_REPORT.md` |
| Original vs cleaned comparison | `research/data_cleaning/ORIGINAL_VS_CLEANED_COMPARISON.md` |
| Cleaned metrics CSV | `research/paper_assets/cleaned_dataset_results.csv` |
| Original dataset snapshot | `dataset_analysis/CrisisMMD_v2.0_original/` |
| Cleaned dataset | `dataset_analysis/CrisisMMD_v2.0_clean/` (images + TSVs + `splits/`) |
| Feature cache (clean) | `backend/outputs/checkpoints/v3_clean/feature_cache.pt` |
| Model weights / summaries / eval | `backend/outputs/checkpoints/v3_clean/` |
| Architecture diagram (PNG + generator) | `research/assets/architecture/` |
| Architecture doc | `research/06_Project_Architecture.md` |

## 6. Integrity check summary

29 automated checks covered artifact existence, sample counts (421 removed / 3,105 kept),
original-copy fidelity (18,104 source images == 18,104 copy images), clean-copy image
membership matching the ledger exactly, split sizes/classes, **zero near-duplicate
leakage**, feature-cache shape/order, and metric-consistency between summary JSON, eval
JSON, and results CSV. **Result: 29/29 PASS.**

## 7. Recommended next steps (not performed)

- If a paper update is ever desired: report the cleaned macro-F1 (0.4829) as the
  leakage-free estimate and disclose the 94-cluster leakage in the original split.
- Consider whether the cleaned, leak-free split should become the default for any future
  training runs.
