# CrisisMind — Deletion Candidates & Archive Plan

**Phase 8 deliverable. NO deletion has been performed. This is the proposed plan awaiting approval.**

## 1. DO NOT TOUCH (required — final report, PPT, code, reproducibility, provenance)

- `research/final_experiment/**` (all phases + FINAL_REPORT)
- `backend/outputs/checkpoints/final_exp/**` (final results, checkpoints, features, grids, seeds)
- `research/optimization/cleaned_dataset.csv` (authoritative split)
- `research/data_cleaning/**` (audit/cleaning/ledgers)
- `research/paper_assets/final/**` (all 7 figures + scripts + provenance)
- `research/paper_assets/{*}.csv` (provenance CSVs)
- `research/Conference/**` (references used)
- `research/references/README.md` + old CSVs (historical record)
- `research/assets/architecture/**` (fig1 + generator)
- `project_audit/**`, `research/00_08 docs`, `IEEE_ASSET_CHECKLIST.md`
- `backend/**` source, `backend/requirements.txt`, `backend/tests/**`
- `frontend/src/**`, `vite.config.ts`, `package*.json`, `tsconfig*`, `index.html`, `public/**`
- `testing/**` docs + sample images (PPT + evidence)
- `dataset_analysis/CrisisMMD_v2.0/**` (min active dataset) + `CrisisMMD_v2.0_clean/**` (cleaned)
- Deployed checkpoints `backend/outputs/checkpoints/best_{model,text_model,gated_fusion_v3,concat_fusion}.pth`
- `dataset_audit.py`, `run_image_baseline.py`, `run_text_baseline.py`, `run_text_eval.py`, `run_fusion.py`

## 2. MOVE TO ARCHIVE (historical experiments — keep for traceability, exclude from final report)

Proposed destination: `research/archive/` (or a sibling `archive/` folder outside the active tree).

| Group | Size | Content | Why archive (not delete) |
|---|---|---|---|
| `research/optimization/P2..P7*` | ~1.7 GB | 60.11%/53.72% grid-search checkpoints + metrics | Historical progression; cited in FINAL_REPORT §11 comparison table |
| `backend/outputs/checkpoints/v3/**` | ~0.6 GB | V3 leaky grid (hundreds of checkpoints) | Old experiment; never present as final |
| `backend/outputs/checkpoints/v3_clean/**` | ~0.1 GB | Leak-free cleaned retrain (0.4829 MF1) | Superseded by final experiment but valid leak-free evidence |
| `backend/outputs/checkpoints/v2/**` | ~0.1 GB | Focal/WCE regularization sweep | Old ERA |
| `backend/outputs/checkpoints/best_gated_fusion*.pth` (v1/v2 only) | 9 MB | Unreferenced old-head checkpoints | Keep 2 copies for traceability |
| `experiments/run_001..005/**` | 24.7 MB | Old fusion runs | Historical |
| `backend/outputs/plots/**`, `metrics/**`, `reports/**` | ~1 MB | Old-era figures/CSVs | Ensure no old figure is confused with final |
| Root-level `FINAL_*_REPORT.md`, `MODEL_*_REPORT.md`, `ROOT_CAUSE_ANALYSIS.md` | 40 KB | July-era reports | Historical record |
| `research/assets/**` (excl architecture) | ~1 MB | Old-era figures referenced by phase docs | Keep so phase docs still resolve |
| `dataset_analysis/CrisisMMD_v2.0_original/**` | ~4 GB | Snapshot duplicate of active dataset | Keep one copy for provenance; it duplicates active `CrisisMMD_v2.0/` |

## 3. SAFE TO DELETE (verified unnecessary — after user approval)

| Item | Size | Verification |
|---|---|---|
| Runtime logs (`backend.log`, `frontend.log`, `test_server.log`, `research/final_experiment/*.log`, `frontend/frontend.log`) | <1 MB | Logs only; not referenced by code |
| `__pycache__/`, `*.pyc` | <5 MB | Regenerable bytecode |
| `.DS_Store`, `._*` (6) | <1 MB | macOS junk |
| `frontend/tsconfig.tsbuildinfo` | 1 KB | TS build cache |
| `research/final_experiment/img.pid` | <1 KB | Stale PID file |
| `backend/outputs/explanations/*` | ~10 MB | Runtime Grad-CAM outputs (regenerable) |
| `backend/outputs/cached_features/*` | ~29 MB | Feature cache (regenerable) |
| `frontend/dist/**` | ~1 MB | Build output (regenerable via npm run build) |
| `backend/.venv/**`, `testing/.venv/**`, `frontend/node_modules/**` | ~0.9 GB | VEs (regenerable) — *defer until after final demo/report run* |

## 4. APPROVAL REQUIRED

**No file has been deleted, moved, renamed, or overwritten in this run.**
Please review and approve each of the three groups above (all / sections / items) before any action.

---
*Audit date: 2026-09-01 · Safety rule respected: a file old is not automatically deletable.*