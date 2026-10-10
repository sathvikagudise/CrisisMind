# CrisisMind — Repository Cleanup Audit Report

**Project:** CrisisMind — Multimodal Deep Learning Framework for Disaster Damage Assessment and Emergency Decision Support

**Date of audit:** 2026-09-01  ·  **Status:** Phase 1 COMPLETE (audit only — nothing deleted)

## 1. Scope & method

- Full recursive scan of the repository root (excluding `node_modules`, `__pycache__`, `.git` internals).
- `FILE_INVENTORY.csv` records every file with path/type/size/last-modified.
- `FILE_CLASSIFICATION.csv` assigns one of 7 categories to every file, plus the purpose, duplicate hint, and a safe-to-delete flag.
- Heavy directory bands (`dataset_analysis/**/data_image/**` images and `.venv`/`node_modules`) are summarised as band rows for readability.

- **Files scanned (detailed):** 1474
- **Files scanned (banded):** 2 in 2 band rows (dataset images + venv/node_modules)
- **Bytes (detailed files):** 5.65 GB

## 2. Classification summary

| Category | Files | Size | Verdict |
|---|---|---|---|
| **REQUIRED FOR FINAL REPORT** | 321 | 1.19 GB | Must never be deleted. Direct evidence for the MU Major Project-I report. |
| **REQUIRED FOR CODE REPRODUCTION** | 214 | 391.77 MB | Must never be deleted. Source code, config, requirements needed to reproduce and deploy the system. |
| **REQUIRED FOR PPT FIGURES SCREENSHOTS** | 76 | 16.81 MB | Must never be deleted. Final figures, sample inputs, screenshot-ready assets for the PPT. |
| **HISTORICAL OLD EXPERIMENT** | 812 | 3.91 GB | Do NOT delete automatically. Move to an archive folder (research/archive/) for traceability; never used as final results. |
| **DUPLICATE REDUNDANT** | 29 | 91.44 MB | Candidates for removal after identity verification. Keep until archive confirmed. |
| **TEMPORARY CACHE LOG** | 22 | 39.98 MB | Safe to delete (regenerable). No reproducibility impact. |
| **UNKNOWN DO NOT DELETE** | 0 | 0 B | Cannot verify purpose. Kept until manually reviewed. |

**Band rows** (dataset images + VEs reproduced separately):
| Band | Files | Size (hosted on disk) | Category |
|---|---|---|---|
| `**/data_image/**` disaster images (7 events) | 39,313 | 4.21 GB | REQUIRED FOR CODE / REPRODUCTION (needed for retraining, Grad-CAM test set, dataset evidence) |
| `.venv/` + `node_modules/` | 27,775 | 0.90 GB | TEMPORARY / CACHE (regenerable via requirements.txt + npm install) |

## 3. Evidence required for the FINAL report (Phase 2 protection list)

| # | Artifact | Location |
|---|---|---|
| 1 | Final test results (single frozen eval) | `backend/outputs/checkpoints/final_exp/FINAL_TEST_RESULTS.json` |
| 2 | Final experiment report | `research/final_experiment/FINAL_REPORT.md` + `.json` |
| 3 | Final model/checkpoint | `backend/outputs/checkpoints/final_exp/F4_generic_wce_bal_n0.0_s42/best.pth` |
| 4 | Feature caches fed to fusion | `backend/outputs/checkpoints/final_exp/features_generic.pt` / `features_dapt.pt` |
| 5 | Dataset split information | `research/optimization/cleaned_dataset.csv` (436 test), `dataset_analysis/CrisisMMD_v2.0_clean/splits/*` |
| 6 | Data cleaning/audit reports | `research/data_cleaning/DATASET_AUDIT_REPORT.md`, `CLEANING_SUMMARY.md`, `CLEANED_DATASET_SPLIT.md` |
| 7 | Kept samples ledger | `research/data_cleaning/KEPT_SAMPLES.csv` |
| 8 | Removed samples ledger | `research/data_cleaning/REMOVED_SAMPLES.csv` |
| 9 | Final training configuration | `FINAL_TEST_RESULTS.json`→`config` (loss=wce, sampling=bal, noise=0, head=softmax, seed=42, lr=1e-4, dropout=0.2, best_epoch=13) |
| 10 | Seed / config provenance | `backend/outputs/checkpoints/final_exp/seed_runs.json` |
| 11 | Confusion matrix source | `FINAL_TEST_RESULTS.json.confusion_matrix` ([[24,19,18],[11,37,51],[8,26,242]]) |
| 12 | Model comparison source | `research/paper_assets/model_comparison.csv` + `ablation_results.json` + `seed_runs.json` |
| 13 | Grad-CAM selections | `research/paper_assets/final/fig5_gradcam_selections.json` + explainability service |
| 14 | Figure provenance | `research/paper_assets/final/FIGURE_PROVENANCE.md` + `FIGURE_MANIFEST.json` |
| 15 | Inference timing evidence | `FINAL_TEST_RESULTS.json` (133.4±17.7 ms full, 1.349±0.172 ms head) |
| 16 | Training/eval scripts | `research/final_experiment/phase0..6*.py`, `backend/training/*` |
| 17 | Architecture diagram generator | `research/assets/architecture/generate_architecture_diagram.py` |
| 18 | Backend inference code | `backend/analysis_engine/*`, `backend/api/*`, `backend/models/fusion_models.py` |
| 19 | Frontend/dashboard code | `frontend/src/**` |
| 20 | Requirements / env files | `backend/requirements.txt`, `frontend/package.json` |
| 21 | README / system docs | `research/README.md`, `backend/README.md`, `frontend/README.md`, project_audit/* |

## 4. Old / invalid result identification (Phase 3)

The following are **HISTORICAL — NOT FOR FINAL REPORT**. They carry the old leaky 60.11% era or pre-cleaning results and must never be presented as final:

- `research/optimization/P2..P7*` grid-search checkpoints/metrics (60.11%/53.72% era; leaky encoders)
- `backend/outputs/checkpoints/v3/` full grid (hundreds of `*_best.pth` + `*_cm.png`, leaky-split era)
- `backend/outputs/checkpoints/v3_clean/` (cleaned-model 0.4829 macro-F1 — legitimate leak-free result but superseded by final experiment)
- `backend/outputs/checkpoints/v2/` (focal/WCE regularization sweep)
- `backend/outputs/checkpoints/best_gated_fusion.pth` / `best_gated_fusion_v2.pth` (v1/v2, unreferenced)
- `experiments/run_001..005` (old fusion runs)
- `backend/outputs/plots/*`, `backend/outputs/metrics/*`, `backend/outputs/reports/*` (old-era figures/metrics)
- `research/assets/*` figures (old-era; some referenced by phase docs — keep in archive)
- Root-level `FINAL_RELEASE_REPORT.md`, `FINAL_VALIDATION_REPORT.md`, `MODEL_AUDIT_REPORT.md`, `MODEL_IMPROVEMENT_REPORT.md`, `ROOT_CAUSE_ANALYSIS.md` (July-era)
- `testing/*.png` (old testing plots), `testing/*_REPORT.md` v2/v3 reports

> **Safety:** none of these were deleted. They are candidates for a move to `research/archive/` (Phase 8 / DELETE_CANDIDATES.md).

## 5. Duplicates identified (Phase 1)

| Duplicate target | Duplicate copy | Detail |
|---|---|---|
| `dataset_analysis/CrisisMMD_v2.0/` (annotations + json + Readme + `data_image/`) | `dataset_analysis/CrisisMMD_v2.0_original/` | Byte-for-byte snapshot copy made for provenance. The active dataset folder already holds the original data. **Move to archive** (kept for provenance) or delete after verification. |
| `dataset_analysis/CrisisMMD_v2.0/annotations/*.tsv` | `dataset_analysis/CrisisMMD_v2.0_original/annotations/*.tsv` | Same files. |
| `dataset_analysis/CrisisMMD_v2.0_original/` annotations+json | `dataset_analysis/CrisisMMD_v2.0/annotations+json` | Identical content. |
| `dataset_analysis/CrisisMMD_v2.0/crisismmd_datasplit_all.zip` | — | Original downloaded zip; contents derived into the folders. Redundant download artifact. |
| `.DS_Store` / `._*` AppleDouble (6 files) | — | macOS junk in dataset folders. |

## 6. Temporary / cache / log files identified

| File(s) | Type | Action |
|---|---|---|
| `.venv/` (backend + testing) | Python venv | Regenerable — delete only when confirmed all libs in requirements.txt |
| `frontend/node_modules/`, `frontend/dist/` | npm deps + build | Regenerable via `npm install` / `npm run build` |
| `backend/outputs/cached_features/**` | Feature cache | Regenerable via feature extraction |
| `backend/outputs/explanations/**` | Runtime Grad-CAM outputs | Regenerable at request time |
| `*.log` (backend.log, frontend.log, test_server.log, research/final_experiment/*.log) | Runtime logs | Delete |
| `__pycache__/`, `*.pyc` | Bytecode cache | Delete |
| `research/final_experiment/img.pid`, `frontend/tsconfig.tsbuildinfo` | PID/build cache | Delete |
| `.DS_Store`, `._*` | macOS junk | Delete |

## 7. Proposed cleanup plan (Phase 8 — see DELETE_CANDIDATES.md)

- **MOVE TO ARCHIVE (~3.9 GB):** old grid-search experiments (`research/optimization/P2..P7`, `backend/outputs/checkpoints/v3`, `v3_clean`, `v2`, `experiments/run_001..005`, old plots/metrics/reports, old root-level reports, `dataset_analysis/CrisisMMD_v2.0_original`).
- **SAFE TO DELETE (~40 MB + 0.9 GB regenerable):** logs, caches, macOS junk, `frontend/dist`, venv/node_modules only after reinstall verification.
- **DO NOT TOUCH:** everything in Section 3 + all source code + final figures + dataset images.

---
*Generated automatically by research/final_cleanup/classify_repo.py + audit_report_gen.py. No files were deleted or modified during this audit.*
## 8. FIGURE AUDIT (Phase 4) — final figure set, all VERIFIED

| File | Size | SHA-256 ([:12]) | Status |
|---|---|---|---|
| `fig1_architecture.png` | 1,602,834 B | `22a8ac27ee4c` | PRESENT |
| `fig2_gated_fusion_v3.png` | 182,649 B | `3bbc118203b9` | PRESENT |
| `fig2_gated_fusion_v3.pdf` | 51,924 B | `35da1141e0b9` | PRESENT |
| `fig2_gated_fusion_v3.svg` | 105,019 B | `875891665c8a` | PRESENT |
| `fig3_model_comparison.png` | 86,068 B | `057de693b1a3` | PRESENT |
| `fig3_model_comparison.pdf` | 24,187 B | `45a02e4d1dd5` | PRESENT |
| `fig3_model_comparison.svg` | 56,300 B | `5a04dcb30d51` | PRESENT |
| `fig4_confusion_matrix.png` | 142,532 B | `92631bae32ae` | PRESENT |
| `fig4_confusion_matrix.pdf` | 28,454 B | `c39f339778aa` | PRESENT |
| `fig4_confusion_matrix.svg` | 68,471 B | `416ee6565d4a` | PRESENT |
| `fig5_gradcam.png` | 1,343,730 B | `7f0b60f3524e` | PRESENT |
| `fig5_gradcam.pdf` | 1,039,232 B | `0ca13e9dc79e` | PRESENT |
| `fig5_gradcam.svg` | 1,592,785 B | `986e0b4015ed` | PRESENT |
| `fig5_gradcam_selections.json` | 1,083 B | `1c06c66db484` | PRESENT |
| `fig6_final_metrics.png` | 63,556 B | `11a29c8627e7` | PRESENT |
| `fig6_final_metrics.pdf` | 18,530 B | `71290395adc4` | PRESENT |
| `fig6_final_metrics.svg` | 38,202 B | `27f71f66a14a` | PRESENT |
| `fig7_classwise_metrics.png` | 52,128 B | `c88a83fdf90f` | PRESENT |
| `fig7_classwise_metrics.pdf` | 16,127 B | `5a8539b95c6f` | PRESENT |
| `fig7_classwise_metrics.svg` | 36,671 B | `d2783696f682` | PRESENT |

- fig1_architecture.png is byte-identical to `research/assets/architecture/fig1.png` (verified earlier).
- All numeric figures source values directly from `backend/outputs/checkpoints/final_exp/FINAL_TEST_RESULTS.json` and `ablation_results.json` (frozen, hashed).
- Generators: `research/paper_assets/final/scripts/*.py`. QC: `scripts/qc_and_manifest.py` re-run passes (fig3 values, fig4 CM, fig6 metrics, fig7 classwise, fig1 byte-identical, no stale constants).
- `fig5_gradcam.png` created with Cam_GradCAM (target layer `backbone.features.8`); selections + confidences in `fig5_gradcam_selections.json`; non-degeneracy independently recomputed (std 0.229–0.267, spread ≥ 0.944, pred==GT).
- `Architecture_IEEE@crisismind.png` (1.4 MB, Aug 27) — alternate IEEE-style architecture PNG held as a PPT candidate.
- Stale-era figures (`backend/outputs/plots/*`, `v3/*_cm.png`, `testing/*.png`, `experiments/run_00X curves`) are NOT part of the final set.

## 9. AI-TOOL / PROMPT AUDIT (Phase 6) — evidence-based, no invention

- **Model components (open-source, non-generative):** EfficientNet-B0 (PyTorch/Timm), DistilBERT (HF Transformers), custom Gated-Fusion head, Grad-CAM, rule-based Decision Engine. These are the model itself, not 'AI assistants'.
- **AI-assisted development (documented in this session):** the OpenCode assistant generated/refined figure scripts, QC harness, provenance docs, Grad-CAM selection, deployment recipes, and this audit. Prompts used are the actual chat requests from this workspace session (visible in the session history).
- **Unknown / cannot verify:** no prompt logs or AI-voucher files exist in the repo. Any claim of specific external-tool prompts must be added by the user (`REPORT_DATA_PACKAGE.md` §X marked NOT VERIFIED).
- **Guardrail:** no AI-authored content is cited as independent research; all numbers trace to repository artifacts.

## 10. FINAL category summary (verified tally, exact CSV values)

| Category | Detailed files | Size |
|---|---|---|
| REQUIRED_FOR_FINAL_REPORT | 321 | 1.19 GB |
| REQUIRED_FOR_CODE_REPRODUCTION | 214 | 391.8 MB |
| REQUIRED_FOR_PPT_FIGURES_SCREENSHOTS | 76 | 16.8 MB |
| HISTORICAL_OLD_EXPERIMENT | 812 | 3.91 GB |
| DUPLICATE_REDUNDANT | 29 | 91.4 MB |
| TEMPORARY_CACHE_LOG | 22 | 40.0 MB |
| UNKNOWN_DO_NOT_DELETE | 0 | 0 |
| **Detailed total** | **1,474** | **~5.6 GB** |
| dataset-images band (`**/data_image/**`, 7 events) | 39,313 | 4.21 GB |
| venv-or-node_modules band (recreatable) | 27,775 | 896.5 MB |
| **Scan total** | **68,562** | **~10.8 GB** |

> No files deleted, moved, renamed, or overwritten during this audit. All tallies derive from `FILE_CLASSIFICATION.csv`.
