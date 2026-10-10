# CrisisMind — IEEE Paper Asset Checklist

> Audit performed on 2026-08-01. This checklist maps every asset required for an
> IEEE-style paper (Methodology, Results, Discussion, Conclusion) to its location
> in the repository and flags which items are READY vs MISSING.
>
> **Method:** filesystem inspection only — no project files were modified, no new
> figures were generated during this audit.

---

## Legend

- **READY** — asset exists on disk in the listed location.
- **MISSING** — asset does not exist anywhere in the repository.
- **PARTIAL** — asset exists but in a different/incomplete form (e.g. mermaid source
  instead of a rendered figure, or generated artifact outside `research/assets`).

---

## 1. Methodology

| Section | Asset | Location | Ready |
|---|---|---|---|
| Dataset | CrisisMMD_v2.0 dataset | `dataset_analysis/CrisisMMD_v2.0/` | YES |
| Dataset | Event-wise distribution figure | `research/assets/dataset/eventwise_distribution.png` | YES |
| Dataset | Humanitarian category distribution | `research/assets/dataset/humanitarian_category_distribution.png` | YES |
| Dataset | Damage severity distribution | `research/assets/dataset/severity_distribution.png` | YES |
| Dataset | Dataset quality analysis (conflicting labels) | `testing/DATASET_QUALITY_ANALYSIS.md` | YES |
| Dataset | Sample images (198 PNGs) | `dataset_analysis/CrisisMMD_v2.0/` | YES |
| Preprocessing | Feature-extraction sequence diagram (mermaid) | `research/06_Project_Architecture.md` | PARTIAL (source only) |
| Architecture | System architecture figure | `research/assets/architecture/` | MISSING (folder empty) |
| Architecture | Pipeline diagram (mermaid) | `research/06_Project_Architecture.md` | PARTIAL (source only) |
| Architecture | Training pipeline diagram (mermaid) | `research/06_Project_Architecture.md` | PARTIAL (source only) |
| Architecture | Future pipeline diagram (mermaid) | `research/06_Project_Architecture.md` | PARTIAL (source only) |
| Models | Image baseline (EfficientNet-B0) metrics | `research/references/*_metrics.csv` | YES |
| Models | Text baseline (DistilBERT) metrics | `research/references/text_metrics.csv` | YES |
| Models | ConcatFusion metrics | `research/references/concat_fusion_metrics.csv` | YES |
| Models | GatedFusion metrics | `research/references/gated_fusion_metrics.csv` | YES |
| Models | GatedFusionWide (v3) — 136 hyperparameter configs | `backend/outputs/checkpoints/v3/` | YES |
| Training | Training curves (all four models) | `research/assets/training_curves/` | YES |
| Hyperparameters | Best v3 hyperparameters | `testing/BEST_HYPERPARAMETERS.md` | YES |
| Training | v3 all-runs F1 comparison graph | `testing/graphs/V3_ALL_RUNS_F1.png` | YES |
| Training | v3 model comparison graph | `testing/graphs/V3_MODEL_COMPARISON.png` | YES |
| Training | v3 gate scatter graph | `testing/graphs/V3_GATE_SCATTER.png` | YES |

---

## 2. Results

| Section | Asset | Location | Ready |
|---|---|---|---|
| Results | Final comparison table (F1, acc, prec, rec) | `research/references/fusion_comparison_report.md` | YES |
| Results | Per-class classification report | `research/references/classification_report.csv` | YES |
| Results | Confusion matrix — image baseline | `research/assets/confusion_matrices/image_baseline_cm.png` | YES |
| Results | Confusion matrix — text baseline | `research/assets/confusion_matrices/text_baseline_cm.png` | YES |
| Results | Confusion matrix — ConcatFusion | `research/assets/confusion_matrices/concat_fusion_cm.png` | YES |
| Results | Confusion matrix — GatedFusion (v1) | `research/assets/confusion_matrices/gated_fusion_cm.png` | YES |
| Results | Confusion matrices — v3 (100+ per config) | `backend/outputs/checkpoints/v3/*_cm.png` | PARTIAL (not in research/assets) |
| Results | Confusion matrices — v2 (per config) | `backend/outputs/checkpoints/v2/*_cm.png` | PARTIAL (not in research/assets) |
| Results | ROC curves per class | — | MISSING |
| Results | Precision-Recall curves per class | — | MISSING |
| Results | Manual evaluation (v1=8/15, v2=9/15, v3=10/15) | `testing/MANUAL_EVALUATION.md` | YES |
| Results | Model comparison (v1 vs v2) graph | `testing/graphs/ALL_RUNS_F1.png`, `testing/graphs/MODEL_COMPARISON.png` | YES |
| Results | Model validation report | `testing/MODEL_VALIDATION_REPORT.md` | YES |

---

## 3. Discussion

| Section | Asset | Location | Ready |
|---|---|---|---|
| Discussion | Gate collapse root cause analysis | `testing/ROOT_CAUSE_ANALYSIS.md`, `ROOT_CAUSE_ANALYSIS.md` | YES |
| Discussion | Gate analysis (v2/v3) | `testing/GATE_ANALYSIS.md` | YES |
| Discussion | Known limitations | `testing/KNOWN_LIMITATIONS.md` | YES |
| Discussion | GradCAM/attention heatmap figures | — | MISSING |
| Discussion | Decision-engine / system flowchart | `research/assets/architecture/` | MISSING (folder empty) |
| Discussion | Ablation insight (gate weight comparison) | `testing/GATE_ANALYSIS.md` | YES |

---

## 4. Conclusion / Meta

| Section | Asset | Location | Ready |
|---|---|---|---|
| Conclusion | Novelty & contribution statement | `research/08_Novelty_and_Contribution.md` | YES |
| Conclusion | Project overview | `research/00_Project_Overview.md`, `FINAL_RELEASE_REPORT.md` | YES |
| Conclusion | References (11 files incl. 6 CSVs) | `research/references/` | YES |
| Conclusion | UI screenshots | `research/assets/screenshots/` | MISSING (folder empty) |
| Conclusion | API test guide (27 endpoints) | `testing/API_TESTS.md` | YES |
| Conclusion | Dashboard audit | `testing/DASHBOARD_AUDIT.md` | YES |
| Conclusion | Landing page audit | `testing/LANDING_PAGE_AUDIT.md` | YES |
| Conclusion | Deployment / environment notes | `testing/KNOWN_LIMITATIONS.md` §6 | YES |

---

## Missing Asset Summary (to generate before submission)

1. **System architecture figure** — `research/assets/architecture/` is empty; only
   mermaid source exists in `research/06_Project_Architecture.md`.
2. **UI screenshots** — `research/assets/screenshots/` is empty; no screenshot exists
   anywhere in the repo (frontend is running for capture).
3. **GradCAM / attention heatmap figures** — none found (feature exists in backend
   but no figure artifacts saved).
4. **ROC and Precision-Recall curves** — none found anywhere in the repo.
5. **v3 confusion matrix in `research/assets`** — only the four v1-era baseline CMs are
   curated; v3 matrices (~136 configs) exist only under `backend/outputs/checkpoints/v3/`.
6. **Decision-engine flowchart figure** — missing; mermaid pipeline diagram only.
7. **Mermaid → rendered image export** — the 4 architecture diagrams exist only as
   `.md` source and need to be exported to PNG/SVG for the paper.

---

## Coverage Summary

| Category | Available | Missing | Coverage |
|---|---|---|---|
| Dataset figures | 3 (+198 samples) | 0 | 100% |
| Training curves | 3 | 0 | 100% |
| Architecture figures | 0 (4 mermaid source) | 1+ | 0% (source only) |
| Confusion matrices (curated) | 4 | 1 (v3) | 80% |
| Comparison graphs | 5 (testing/graphs) | 0 | 100% |
| UI screenshots | 0 | 1+ | 0% |
| GradCAM / ROC / PR figures | 0 | 3 | 0% |
| Metrics / reports | 10 research md + ~15 testing md | 0 | 100% |
| Tables in research md | ~135 across 10 files | — | — |
