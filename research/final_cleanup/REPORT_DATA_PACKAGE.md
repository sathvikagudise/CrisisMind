# CrisisMind — Final Report Data Package

**Project title:** CrisisMind — Multimodal Deep Learning Framework for Disaster Damage Severity Assessment and Emergency Decision Support
**Purpose:** Single-source-of-truth data pack for the MU Major Project-I final report and PPT.
**Status:** Phase 5 deliverable. Every value below was read directly from authoritative artifacts listed in the "Source" column. Anything not verifiable in the repository is explicitly marked **NOT VERIFIED — NEEDS USER INPUT**.

---

## A. Project identity
| Field | Value | Source / status |
|---|---|---|
| Project name | CrisisMind — Multimodal Disaster Damage Severity Assessment | `research/00_Project_Overview.md` |
| Proposed by | **(student team members)** | **NOT VERIFIED — NEEDS USER INPUT** |
| Guided by | **(guide name & department)** | **NOT VERIFIED — NEEDS USER INPUT** |
| University | Mumbai University (MU) | User-provided context |
| Academic year / term | **(final Major Project-I)** | **NOT VERIFIED — NEEDS USER INPUT** |

## B. Problem statement / aim
- Aim: automatic severity classification (little or no / mild / severe damage) of disaster-related images **combined with** their tweet text, addressing data scarcity/ambiguity and naive early-fusion limits of prior work.
- Overall objective: image-only vs text-only vs multimodal Gated fusion comparison, plus Grad-CAM explainability and a production dashboard.
- Source: `research/00_Project_Overview.md`, `research/01-03` docs. *(Original wording should be quoted verbatim in the report; the docs are at `research/00_..08_*.md`.)*

## C. Objectives / scope
1. Build a real multimodal disaster corpus from CrisisMMD with validated severity labels.
2. Implement deep encoders (EfficientNet-B0 image + DistilBERT text) and a Gated-Fusion decision head.
3. Conduct a rigorous leakage-free evaluation (train/val/test disjoint at near-duplicate level).
4. Provide explainability (Grad-CAM) and a working web dashboard.
5. Present ablations (image/text/concat/v1/v2/F4-gated) proving the final design choice.
- Source: `research/00_Project_Overview.md` (objectives) + final configs. *(Report's own sections may re-word; the raw objective list is in that file.)*

## D. Dataset overview
| Item | Value | Source |
|---|---|---|
| Source | CrisisMMD v2.0 (7 events: hurricane_harvey, hurricane_maria, hurricane_irma, california_wildfires, mexico_earthquake, iraq_iran_earthquake, srilanka_floods) | `dataset_analysis/CrisisMMD_v2.0/Readme.txt`, `DATASET_AUDIT_REPORT.md` |
| Full run | Disaster images 16,870 + non-disaster 12,780 (29,650 claimed in readme page); our severity corpus | `research/data_cleaning/DATASET_AUDIT_REPORT.md` |
| Severity-annotated (our subset) | 3,526 (severe 2,212 / mild 839 / little_or_no 475); labels "dont_know"/"cant_judge" excluded | `DATASET_AUDIT_REPORT.md` §severity population |
| After ambiguity/exact-dup/weak-annotator cleaning | 3,105 (severe 2,013 / mild 689 / little_or_no 403) | `CLEANING_SUMMARY.md` |
| Final split (leak-free, seed 42) | train 2,521 / val 460 / test 436 | `research/final_experiment/FINAL_REPORT.md` §6, `optimization/cleaned_dataset.csv` |
| Test-class distribution | little 61 / mild 99 / severe 276 (n=436) | `FINAL_TEST_RESULTS.json` `per_class` support |

## E. Data cleaning decisions (evidence)
| Rule | Rule detail | Removed |
|---|---|---|
| A–E | (phase-doc rules; none triggered in final set) | 0 |
| F | exact-duplicate image removal (subset replicated scores) | 83 |
| G | conflicting near-duplicate pairs removed | 25 |
| H | weak annotator agreement (<2/3 majority), ambiguous labels | 313 |
| **Total removed** | from 3,526 → 3,105 | **421** |
- Plus 312 train-only images reclaimed after leakage fix (2,173 → 2,521 train).
- Sources: `CLEANING_SUMMARY.md`, `CLEANED_DATASET_SPLIT.md`, `FINAL_REPORT.md` §5.

## F. Leakage remediation (evidence)
| Item | Value |
|---|---|
| Test images carrying inherited leakage before fix | 308 / 436 (70.6%) |
| Method | duplicate-image hashing (dHash, Hamming ≤ 0.10), stratified 70/15/15, random seed 42 |
| Near-dup clusters (99) / samples moved (220); 2,885 singletons; 2,984 groups | `CLEANED_DATASET_SPLIT.md` |
| Result | test set visually disjoint from train at 0.10 Hamming threshold |

## G. Model architecture (final)
- **Image encoder:** EfficientNet-B0 CNN (frozen, 1280-d), input 224×224 RGB.
- **Text encoder:** DistilBERT (frozen, 768-d), max sequence length 64 tokens.
- **Fusion head:** Gated Fusion v3 (`GatedFusionWide`: hidden 1024 + BatchNorm) — learnable per-modal gates (target 0.5, gate-reg weight 1.0) — 3,152,645 trainable params.
- **Head:** Softmax over 3 severities.
- Sources: `research/final_experiment/*` trainer logs, `CLEANED_MODEL_REPORT.md` (architecture identical), `backend/models/fusion_models.py`.

## H. Training configuration (final experiment)
| Hyperparameter | Value |
|---|---|
| Loss | Weighted Cross-Entropy (inverse-frequency, normalized) class weights little 1.6805 / mild 0.9832 / severe 0.3363 |
| Sampling | WeightedRandomSampler (inverse-freq, replacement), balanced batches |
| Optimizer / LR / decay | AdamW, 1e-4, weight decay 1e-4 |
| Scheduler | warmup_cosine, 5-epoch warmup |
| Epochs (early stop @ best) | 13 (patience 10) |
| Batch train/val/test | 32 / 64 / 64 |
| Feature noise | 0.0 |
| Head / seed | softmax / 42 |
| Dropout (head) | 0.2 |
| Hardware | CPU (AMD64, no CUDA) |
- Source: `FINAL_TEST_RESULTS.json` `config`.

## I. Final test results (leak-free test, n=436)
| Metric | Value |
|---|---|
| Accuracy | **69.50%** |
| Precision (macro / weighted) | 59.58% / 67.31% |
| Recall (macro / weighted) | 54.80% / 69.50% |
| **Macro F1** | **56.50%** |
| Weighted F1 | 67.94% |
- Source: `FINAL_TEST_RESULTS.json` `test_metrics` / `config.test`.

## J. Per-class results (n=436)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| little_or_no_damage | 55.81% | 39.34% | 46.15% | 61 |
| mild_damage | 45.12% | 37.37% | 40.88% | 99 |
| severe_damage | 77.81% | 87.68% | 82.45% | 276 |
- Source: `FINAL_TEST_RESULTS.json` `per_class`.

## K. Confusion matrix (rows=actual, cols=pred, little/mild/severe)
```
[[ 24  19  18]
 [ 11  37  51]
 [  8  26 242]]
```
- Verified consistency: row sums = supports (61/99/276). Source: `FINAL_TEST_RESULTS.json` `confusion_matrix`.

## L. Inference latency (CPU, steady-state)
| Component | ms |
|---|---|
| Full pipeline (image+text→severity) | 133.38 ± 17.67 |
| Fusion head only | 1.349 ± 0.172 |
- Source: `FINAL_TEST_RESULTS.json` `*_latency_ms_*`.

## M. Ablation study (final experiment, leak-free test set)
| Variant | Acc | Macro-F1 |
|---|---|---|
| Image only (EfficientNet-B0 frozen) | 63.04% | 55.16% |
| Text only (DistilBERT frozen) | 51.30% | 42.64% |
| Concatenation fusion | 58.48% | 51.48% |
| Gated Fusion v1 | 55.00% | 50.98% |
| Gated Fusion v2 | 55.43% | 51.32% |
| **Gated Fusion v3 (final)** | **69.50%** | **56.50%** |
- Source: `backend/outputs/checkpoints/final_exp/ablation_results.json` + `FINAL_TEST_RESULTS.json`.

## N. Historical / NOT-for-final results (context-only)
| Model | Split | Acc | Macro-F1 | Note |
|---|---|---|---|---|
| v3 grid (original, before leak-fix audit) | 529/529 era | 60.11% | 53.72% | Obsolete — leaky encoders/split |
| Gated fusion v3 pre-refactor release | 529-era | 63.71% / 63.52% | — | Obsolete |
| Cleaned-dataset Gated Fusion v3 (`wide_wce_g1.0`) | leak-free 2,173/467/465 | 56.99% | 48.29% | Superseded by final exp (F4) |
| optimization P6 (frozen-enc) | leaky | 63.07% | 55.43% | Historical progression (FINAL_REPORT §11) |
- **These must never be labelled "final".** Sources: `ORIGINAL_VS_CLEANED_COMPARISON.md`, `CLEANED_MODEL_REPORT.md`, `research/optimization/FINAL_REPORT.json`.

## O. Final figure set (Phase 4 audit — all verified)
| Figure | File (research/paper_assets/final/) | Source record |
|---|---|---|
| <fig1> Architecture diagram | `fig1_architecture.png` (+ `.pdf/.svg`) | `FIGURE_PROVENANCE.md` id=1; byte-identical to `research/assets/architecture/fig1.png` |
| <fig2> Gated Fusion v3 block diagram | `fig2_gated_fusion_v3.{png,pdf,svg}` | FIGURE_MANIFEST id=2 |
| <fig3> Model comparison (3 modalities + final) | `fig3_model_comparison.{png,pdf,svg}` | manifest id=3; ablation table (M) |
| <fig4> Confusion matrix (final test) | `fig4_confusion_matrix.{png,pdf,svg}` | manifest id=4; values = table (K) |
| <fig5> Grad-CAM explainability (3 examples) | `fig5_gradcam.{png,pdf}` (SVG may be empty) | manifest id=5; `fig5_gradcam_selections.json` |
| <fig6> Final metrics dashboard | `fig6_final_metrics.{png,pdf,svg}` | manifest id=6; values = table (I) |
| <fig7> Classwise metrics | `fig7_classwise_metrics.{png,pdf,svg}` | manifest id=7; values = table (J) |
- All generated by `research/paper_assets/final/scripts/*.py` from frozen authoritative JSONs; FIGURE_MANIFEST.json holds per-file hash + dimensions; QC probe verified byte-identical re-generation, confusion-matrix/stale-constant checks.

## P. Grad-CAM selections (fig5 source examples)
| GT class | Test row | Conf | Image |
|---|---|---|---|
| Severe Damage | 0 | 99.9% | `dataset_analysis/CrisisMMD_v2.0/data_image/california_wildfires/10_10_2017/917792930315821057_0.jpg` |
| Mild Damage | 50 | 94.1% | `dataset_analysis/CrisisMMD_v2.0/data_image/hurricane_harvey/27_8_2017/901650894151979008_0.jpg` |
| Little or No Damage | 53 | 100.0% | `dataset_analysis/CrisisMMD_v2.0/data_image/hurricane_harvey/27_8_2017/901688029521334272_2.jpg` |
- Source: `research/paper_assets/final/fig5_gradcam_selections.json`.

## Q. Encoders & checkpoints used (final exp)
| Component | Final checkpoint |
|---|---|
| Image encoder (EfficientNet-B0) | `backend/outputs/checkpoints/final_exp/best_model.pth` (frozen) |
| Text encoder (DistilBERT) | `backend/outputs/checkpoints/final_exp/best_text_model.pth` (frozen) |
| Gated fusion head (final) | `backend/outputs/checkpoints/final_exp/F4_generic_wce_bal_n0.0_s42/best.pth` |
| Gated fusion head (deployed app) | `backend/outputs/checkpoints/best_gated_fusion_v3.pth` — **NOTE:** the running dashboard weights differ from the report's F4 head; deployment guide documents this. |

## R. Reproducibility requirements
- Python 3.11 (CPU), torch + transformers + timm + PIL + numpy/pandas + matplotlib (+ FastAPI family for API). See `backend/requirements.txt`.
- Run sequence: feature extraction (`features_generic.pt`/features_dapt), train head with config in (H), evaluate with `FINAL_TEST_RESULTS.json` config, regenerate figures via `research/paper_assets/final/scripts/*.py`.
- Sources: `research/final_experiment/phase*.py`, `FINAL_REPORT.md` §reproducibility.

## S. Key performance claim (one-liner for PPT)
> "Gated multimodal fusion reaches 69.50% accuracy / 56.50% macro-F1 on a leakage-free 436-image CrisisMMD test set with 133 ms CPU inference, plus Grad-CAM explainability and a live dashboard."

## T. System / deployment essentials (Phase 7 — see DEPLOYMENT_GUIDE.md)
- 5 components loaded at startup (image, text, gated-fusion, concat-fusion, classifier) — `GET /health` → `{"status":"healthy","models_loaded":5}`.
- 29 API routes incl. `/predict`, `/explainability/gradcam`, `/disasters/events`, `/batch/upload*`, `/dashboard/*`, `/report/*`, `/analysis-center/*`.
- Frontend Vite (React 19 + TS) on :5173 proxying to :8000.

## U. Apps / dashboard screenshots
- `research/assets/screenshots/` exists but is **empty** — no screenshots captured yet. **Action needed:** capture landing page, upload+prediction, gallery/CAM, dashboard charts before PPT (marked NOT VERIFIED until then).

## V. Reference list (for report bibliography)
1. Firoj Alam et al., CrisisMMD: Multimodal Twitter Datasets from Natural Disasters (ICWSM 2018 / arXiv:1805.00713)
2. Muhammad Imran et al. (ISCRAM / social-media disaster keyword works) — as cited in `research/references/README.md`
3. EfficientNet-B0 (Tan & Le, ICML 2019)
4. DistilBERT (Sanh et al., 2019)
5. Gated Multimodal Units (Arevalo et al., 2020) — gating mechanism
6. Kiela et al., Rethinking Image Representation Aids... (E2E multimodal learning)
7. Imran et al., Surveys on social-media disaster analysis (as cited)
- Full list with browsing details: `research/references/README.md`.

## W. Team & supervision details
| Field | Value | Status |
|---|---|---|
| Student names / roll numbers | — | **NOT VERIFIED — NEEDS USER INPUT** |
| Guide / department | — | **NOT VERIFIED — NEEDS USER INPUT** |
| College | — | **NOT VERIFIED — NEEDS USER INPUT** |

## X. AI tooling disclosure (Phase 6)
- **Vouchers required by MU for AI use** — **NOT VERIFIED — NEEDS USER INPUT** (report section left flagged).
- Model components used (open-source, non-generative): EfficientNet-B0 (Pytorch/Timm), DistilBERT (HuggingFace), custom Gated-Fusion head, Grad-CAM, Decision Engine.
- Development AI-assistance: OpenAI/GPT + OpenCode assisted with code generation, figure scripts, QC, and this audit. **No AI-generated content cited as author-less; any unmodified AI-completed figures/scripts are ours to re-derive.** (Prompts/evidence of AI use are not stored in the repo; add a short disclosure paragraph per university policy.)

## Y–Z. File-standard metadata
| Field | Value |
|---|---|
| Report title (suggested) | "CrisisMind: Multimodal Deep Learning for Disaster Damage Severity Assessment and Decision Support" |
| Expected sections | Intro, Background, Dataset & Preprocessing, System Design, Experiments & Results, Explainability, Conclusion & Future Work, References, Appendix |
| IEEE paper length | ~6–8 pp (user request earlier) |

## AA–AJ. Open items (to close before submission)
| ID | Item | Status |
|---|---|---|
| AA | Team member names & roll numbers | **NEEDS USER INPUT** |
| AB | Guide name & department | **NEEDS USER INPUT** |
| AC | Screenshots of app (landing/predict/gallery/CAM/dashboard) | **TODO — folders empty** |
| AD | MU vouchers / page-limit calibration | **NEEDS USER INPUT** |
| AE | Verify fusion-head weights parity deployed (v3) vs report (F4) before demo claims | Documented gap (Q) |
| AF | Optional: rerun final figures if any source JSON is regenerated | Guard by QC probe |
| AG | Decide archive location for old experiments (DELETE_CANDIDATES.md §2) | Pending approval |

---

*Generated 2026-09-01 from repository artifacts; every numeric claim cites its source file. Items marked NOT VERIFIED must not be quoted as fact.*