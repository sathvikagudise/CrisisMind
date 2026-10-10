# CrisisMind — Final IEEE Paper Figure Provenance

All quantitative figures are generated **directly from the authoritative artifacts** of the final leakage-free experiment (`backend/outputs/checkpoints/final_exp/FINAL_TEST_RESULTS.json`, `ablation_results.json`, `seed_runs.json`). No values were invented, estimated, interpolated, or copied from earlier experiments.

**Final configuration:** `F4_generic_wce_bal_n0.0_s42` — frozen ImageNet EfficientNet-B0 + frozen vanilla DistilBERT, WCE + balanced sampling, lr 1e-4, dropout 0.2, softmax head, seed 42, test n=436.

## Per-figure provenance

| Figure | Source / method | Data values used | Status |
|---|---|---|---|
| fig1_architecture.png | Copy (byte-identical, sha256-verified) of the approved `research/assets/architecture/fig1.png`. Architecture/schematic only — no metric values plotted. | N/A (schematic) | Final ✓ |
| fig2_gated_fusion_v3.png | Generated diagram faithful to the actual v3 trunk (`backend/models/fusion_models.py::GatedFusionWide`; identical in `phase4_search.py` used by the final checkpoint). Joint gate MLP over the concatenated features; scalar gates; gated concat → FC1024-BN-ReLU-Drop → FC512-BN-ReLU-Drop → FC3 → Softmax. | Architecture equations only | Final ✓ |
| fig3_model_comparison.png | Accuracy & Macro-F1 per model, read live from `ablation_results.json` (unimodal/concat/v1/v2) and `seed_runs.json` (v3 = frozen seed-42 test eval). Asserted equal to FINAL_REPORT.md values. | 64.68/53.35 · 53.67/46.25 · 64.68/56.18 · 58.72/54.05 · 58.72/54.05 · **69.50/56.50** (Acc/F1m %) | Final ✓ |
| fig4_confusion_matrix.png | Confusion matrix read live from `FINAL_TEST_RESULTS.json.confusion_matrix`; row-normalised recall annotated. | [[24,19,18],[11,37,51],[8,26,242]], supports 61/99/276, n=436 | Final ✓ |
| fig5_gradcam.png | REAL Grad-CAM computed at generation time with the project's `backend/explainability/gradcam.py` targeting EfficientNet-B0 `features.8`, on actual cleaned-split test images through the final checkpoint `F4_generic_wce_bal_n0.0_s42/best.pth` (frozen generic encoders + fusion head). Deterministic selection: first correctly-predicted test row per severity class. GT/pred/confidence shown are the genuine pipeline outputs (see `fig5_gradcam_selections.json`). | computed live; see selections JSON | Final ✓ |
| fig6_final_metrics.png | Five headline metrics read live from `FINAL_TEST_RESULTS.json.test_metrics`; asserted against FINAL_REPORT.md. | 69.50 / 59.58 / 54.80 / 56.50 / 67.94 (%) | Final ✓ |
| fig7_classwise_metrics.png | Per-class Precision/Recall/F1 read live from `FINAL_TEST_RESULTS.json.per_class`; asserted against FINAL_REPORT.md. | L/N: 55.81/39.34/46.15 · Mild: 45.12/37.37/40.88 · Severe: 77.81/87.68/82.45 | Final ✓ |

## Generation scripts

Editable sources for every figure live in `research/paper_assets/final/scripts/`:
- `scripts/fig2_gated_fusion_v3.py`
- `scripts/fig3_model_comparison.py`
- `scripts/fig4_confusion_matrix.py`
- `scripts/fig5_gradcam.py`
- `scripts/fig6_final_metrics.py`
- `scripts/fig7_classwise_metrics.py`
- `scripts/qc_and_manifest.py`
- `scripts/style.py`

Each quantitative script contains hard assertions that fail the build if a plotted value ever drifts from the authoritative JSON artifacts.

## QC verification results

| Check | Result |
|---|---|
| fig3 vs expected ablation values | PASS |
| fig4 CM == [[24,19,18],[11,37,51],[8,26,242]] and n=436 | PASS |
| fig6 five metrics match FINAL_REPORT.md | PASS |
| fig7 classwise P/R/F1 match FINAL_REPORT.md | PASS |
| fig1 is byte-identical copy of approved architecture figure | PASS |
| no old-experiment constants in generator scripts | PASS |

| File | Exists | Format | Size (px) | DPI | Bytes |
|---|---|---|---|---|---|
| fig1_architecture.png | True | PNG | 1536×1024 | - | 1602834 |
| fig2_gated_fusion_v3.png | True | PNG | 1926×898 | (299.9994, 299.9994) | 182649 |
| fig3_model_comparison.png | True | PNG | 1923×877 | (299.9994, 299.9994) | 86068 |
| fig4_confusion_matrix.png | True | PNG | 1515×1188 | (299.9994, 299.9994) | 142532 |
| fig5_gradcam.png | True | PNG | 2050×1823 | (299.9994, 299.9994) | 1343730 |
| fig6_final_metrics.png | True | PNG | 1258×841 | (299.9994, 299.9994) | 63556 |
| fig7_classwise_metrics.png | True | PNG | 1280×807 | (299.9994, 299.9994) | 52128 |

**Overall QC verdict: ALL CHECKS PASSED**

## IEEE print-suitability notes

- Raster PNGs exported at 300 DPI; vector PDF/SVG companions provided for every matplotlib-generated figure (fig1 remains the approved raster).
- Single-column figures sized ≈3.5–4.6 in wide; full-width figures (fig2 architecture, fig3 comparison, fig5 Grad-CAM grid) sized ≈6.6–7.16 in.
- Minimum font size used ≈7.4 pt at final scale (≥ 6 pt IEEE floor); axis labels ≥ 8 pt.
- fig4 colourbar and cell text remain readable under greyscale printing (counts printed in every cell).
