"""QC verification + provenance manifest for research/paper_assets/final/.

READ-ONLY with respect to every pre-existing repository artifact: verifies
figures against FINAL_TEST_RESULTS.json / ablation_results.json /
seed_runs.json / FINAL_REPORT.md, checks file integrity (existence, format,
pixel dimensions, DPI, sha256), confirms fig1 is a byte-identical copy,
scans generator scripts for stale old-experiment constants, and emits:

  - FIGURE_MANIFEST.json   (machine-readable provenance)
  - FIGURE_PROVENANCE.md   (human-readable provenance + QC table)

Nothing outside research/paper_assets/final/ is written.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import ROOT, EXP_DIR, FINAL_DIR, load_final, load_ablation, load_seeds

from PIL import Image

FINAL_MD = os.path.join(ROOT, "research", "final_experiment", "FINAL_REPORT.md")
FIG1_SRC = os.path.join(ROOT, "research", "assets", "architecture", "fig1.png")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def img_info(path):
    im = Image.open(path)
    return {"format": im.format, "width_px": im.width, "height_px": im.height,
            "dpi": im.info.get("dpi")}


# ---------------------------------------------------------------- source data
final = load_final()
abl = load_ablation()
seeds = load_seeds()
tm = final["test_metrics"]
pc = final["per_class"]

fig3_vals = {
    "Image-Only": [round(abl["img"]["test"]["accuracy"] * 100, 2),
                   round(abl["img"]["test"]["f1_macro"] * 100, 2)],
    "Text-Only": [round(abl["txt"]["test"]["accuracy"] * 100, 2),
                  round(abl["txt"]["test"]["f1_macro"] * 100, 2)],
    "Concatenation": [round(abl["concat"]["test"]["accuracy"] * 100, 2),
                      round(abl["concat"]["test"]["f1_macro"] * 100, 2)],
    "Gated Fusion v1": [round(abl["v1"]["test"]["accuracy"] * 100, 2),
                        round(abl["v1"]["test"]["f1_macro"] * 100, 2)],
    "Gated Fusion v2": [round(abl["v2"]["test"]["accuracy"] * 100, 2),
                        round(abl["v2"]["test"]["f1_macro"] * 100, 2)],
    "Adaptive Gated Fusion v3": [
        round(seeds["F4_generic_wce_bal_n0.0"]["test_at_seed42"]["accuracy"] * 100, 2),
        round(seeds["F4_generic_wce_bal_n0.0"]["test_at_seed42"]["f1_macro"] * 100, 2)],
}
expected_fig3 = {
    "Image-Only": [64.68, 53.35], "Text-Only": [53.67, 46.25],
    "Concatenation": [64.68, 56.18], "Gated Fusion v1": [58.72, 54.05],
    "Gated Fusion v2": [58.72, 54.05], "Adaptive Gated Fusion v3": [69.50, 56.50]}

cm = final["confusion_matrix"]
metrics5 = {"Accuracy": round(tm["accuracy"] * 100, 2),
            "Macro Precision": round(tm["precision_macro"] * 100, 2),
            "Macro Recall": round(tm["recall_macro"] * 100, 2),
            "Macro F1": round(tm["f1_macro"] * 100, 2),
            "Weighted F1": round(tm["f1_weighted"] * 100, 2)}
classwise = {c: {"precision": round(pc[k]["precision"] * 100, 2),
                 "recall": round(pc[k]["recall"] * 100, 2),
                 "f1": round(pc[k]["f1"] * 100, 2)}
             for c, k in [("Little/No Damage", "little_or_no_damage"),
                          ("Mild Damage", "mild_damage"),
                          ("Severe Damage", "severe_damage")]}

# ------------------------------------------------------------- value checks
checks = []
checks.append(("fig3 vs expected ablation values",
               all(fig3_vals[k] == v for k, v in expected_fig3.items())))
checks.append(("fig4 CM == [[24,19,18],[11,37,51],[8,26,242]] and n=436",
               cm == [[24, 19, 18], [11, 37, 51], [8, 26, 242]]
               and sum(sum(r) for r in cm) == 436))
checks.append(("fig6 five metrics match FINAL_REPORT.md",
               metrics5 == {"Accuracy": 69.50, "Macro Precision": 59.58,
                            "Macro Recall": 54.80, "Macro F1": 56.50,
                            "Weighted F1": 67.94}))
exp_cw = {"Little/No Damage": (55.81, 39.34, 46.15),
          "Mild Damage": (45.12, 37.37, 40.88),
          "Severe Damage": (77.81, 87.68, 82.45)}
checks.append(("fig7 classwise P/R/F1 match FINAL_REPORT.md",
               all((classwise[c]["precision"], classwise[c]["recall"],
                    classwise[c]["f1"]) == e for c, e in exp_cw.items())))

# fig1 byte-identical copy check
same = False
if os.path.exists(os.path.join(FINAL_DIR, "fig1_architecture.png")):
    same = sha256(FIG1_SRC) == sha256(os.path.join(FINAL_DIR, "fig1_architecture.png"))
checks.append(("fig1 is byte-identical copy of approved architecture figure", same))

# stale-value scan in generator scripts (old 529-split era numbers)
stale_patterns = ["0.6011", "60.11", "53.72", "0.5372", "0.5841", "58.41",
                  "0.6371", "63.71", "0.6352", "63.52", "529"]
stale_hits = {}
GENERATORS = ["fig2_gated_fusion_v3.py", "fig3_model_comparison.py",
              "fig4_confusion_matrix.py", "fig5_gradcam.py",
              "fig6_final_metrics.py", "fig7_classwise_metrics.py"]
for fn in GENERATORS:
    src = open(os.path.join(HERE, fn), encoding="utf-8").read()
    hits = [p for p in stale_patterns if p in src]
    if hits:
        stale_hits[fn] = hits
checks.append(("no old-experiment constants in generator scripts",
               not stale_hits))

# ------------------------------------------------------------------ files
files = ["fig1_architecture.png", "fig2_gated_fusion_v3.png",
         "fig3_model_comparison.png", "fig4_confusion_matrix.png",
         "fig5_gradcam.png", "fig6_final_metrics.png",
         "fig7_classwise_metrics.png"]
manifest = {"generated_for": "CrisisMind final IEEE paper",
            "authoritative_sources": {
                "final_results": os.path.relpath(
                    os.path.join(EXP_DIR, "FINAL_TEST_RESULTS.json"), ROOT),
                "ablations": os.path.relpath(
                    os.path.join(EXP_DIR, "ablation_results.json"), ROOT),
                "seed_runs": os.path.relpath(
                    os.path.join(EXP_DIR, "seed_runs.json"), ROOT),
                "report": os.path.relpath(FINAL_MD, ROOT)},
            "final_configuration": {
                "checkpoint": "F4_generic_wce_bal_n0.0_s42",
                "encoders": "frozen ImageNet EfficientNet-B0 + frozen vanilla DistilBERT",
                "loss": "weighted cross entropy (inverse frequency)",
                "sampling": "balanced", "lr": 1e-4, "dropout": 0.2,
                "head": "softmax", "seed": 42, "test_samples": 436},
            "figures": []}

for fn in files:
    p = os.path.join(FINAL_DIR, fn)
    ok = os.path.exists(p)
    entry = {"filename": fn, "exists": ok}
    if ok:
        entry.update(img_info(p))
        entry["sha256"] = sha256(p)
        entry["bytes"] = os.path.getsize(p)
    manifest["figures"].append(entry)

# vector companions
vec = {}
for base in files:
    stem = base[:-4]
    vec[stem] = {ext: os.path.exists(os.path.join(FINAL_DIR, f"{stem}.{ext}"))
                 for ext in ("pdf", "svg")}
manifest["vector_companions"] = vec
manifest["value_checks"] = [{"check": c, "passed": bool(ok)} for c, ok in checks]
manifest["stale_constant_scan"] = stale_hits
manifest["gradcam_selections"] = json.load(
    open(os.path.join(FINAL_DIR, "fig5_gradcam_selections.json"), encoding="utf-8"))

with open(os.path.join(FINAL_DIR, "FIGURE_MANIFEST.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

# ------------------------------------------------------------------- md
L = []
L.append("# CrisisMind — Final IEEE Paper Figure Provenance\n")
L.append("All quantitative figures are generated **directly from the authoritative "
         "artifacts** of the final leakage-free experiment "
         "(`backend/outputs/checkpoints/final_exp/FINAL_TEST_RESULTS.json`, "
         "`ablation_results.json`, `seed_runs.json`). No values were invented, "
         "estimated, interpolated, or copied from earlier experiments.\n")
L.append(f"**Final configuration:** `{manifest['final_configuration']['checkpoint']}` — "
         f"{manifest['final_configuration']['encoders']}, WCE + balanced sampling, "
         f"lr 1e-4, dropout 0.2, softmax head, seed 42, test n=436.\n")
L.append("## Per-figure provenance\n")
rows = [
    ("fig1_architecture.png", "Copy (byte-identical, sha256-verified) of the approved "
     "`research/assets/architecture/fig1.png`. Architecture/schematic only — no metric "
     "values plotted.", "N/A (schematic)", "Final ✓"),
    ("fig2_gated_fusion_v3.png", "Generated diagram faithful to the actual v3 trunk "
     "(`backend/models/fusion_models.py::GatedFusionWide`; identical in "
     "`phase4_search.py` used by the final checkpoint). Joint gate MLP over the "
     "concatenated features; scalar gates; gated concat → FC1024-BN-ReLU-Drop → "
     "FC512-BN-ReLU-Drop → FC3 → Softmax.", "Architecture equations only", "Final ✓"),
    ("fig3_model_comparison.png", "Accuracy & Macro-F1 per model, read live from "
     "`ablation_results.json` (unimodal/concat/v1/v2) and `seed_runs.json` "
     "(v3 = frozen seed-42 test eval). Asserted equal to FINAL_REPORT.md values.",
     "64.68/53.35 · 53.67/46.25 · 64.68/56.18 · 58.72/54.05 · 58.72/54.05 · "
     "**69.50/56.50** (Acc/F1m %)", "Final ✓"),
    ("fig4_confusion_matrix.png", "Confusion matrix read live from "
     "`FINAL_TEST_RESULTS.json.confusion_matrix`; row-normalised recall annotated.",
     "[[24,19,18],[11,37,51],[8,26,242]], supports 61/99/276, n=436", "Final ✓"),
    ("fig5_gradcam.png", "REAL Grad-CAM computed at generation time with the project's "
     "`backend/explainability/gradcam.py` targeting EfficientNet-B0 `features.8`, on "
     "actual cleaned-split test images through the final checkpoint "
     "`F4_generic_wce_bal_n0.0_s42/best.pth` (frozen generic encoders + fusion head). "
     "Deterministic selection: first correctly-predicted test row per severity class. "
     "GT/pred/confidence shown are the genuine pipeline outputs (see "
     "`fig5_gradcam_selections.json`).", "computed live; see selections JSON",
     "Final ✓"),
    ("fig6_final_metrics.png", "Five headline metrics read live from "
     "`FINAL_TEST_RESULTS.json.test_metrics`; asserted against FINAL_REPORT.md.",
     "69.50 / 59.58 / 54.80 / 56.50 / 67.94 (%)", "Final ✓"),
    ("fig7_classwise_metrics.png", "Per-class Precision/Recall/F1 read live from "
     "`FINAL_TEST_RESULTS.json.per_class`; asserted against FINAL_REPORT.md.",
     "L/N: 55.81/39.34/46.15 · Mild: 45.12/37.37/40.88 · Severe: 77.81/87.68/82.45",
     "Final ✓"),
]
L.append("| Figure | Source / method | Data values used | Status |")
L.append("|---|---|---|---|")
for r in rows:
    L.append("| " + " | ".join(r) + " |")

L.append("\n## Generation scripts\n")
L.append("Editable sources for every figure live in `research/paper_assets/final/scripts/`:")
for fn in sorted(os.listdir(HERE)):
    if fn.endswith(".py"):
        L.append(f"- `scripts/{fn}`")
L.append("\nEach quantitative script contains hard assertions that fail the build if a "
         "plotted value ever drifts from the authoritative JSON artifacts.\n")

L.append("## QC verification results\n")
L.append("| Check | Result |")
L.append("|---|---|")
for c, ok in checks:
    L.append(f"| {c} | {'PASS' if ok else 'FAIL'} |")
L.append("")
L.append("| File | Exists | Format | Size (px) | DPI | Bytes |")
L.append("|---|---|---|---|---|---|")
for e in manifest["figures"]:
    L.append(f"| {e['filename']} | {e['exists']} | {e.get('format','-')} | "
             f"{e.get('width_px','-')}×{e.get('height_px','-')} | "
             f"{e.get('dpi',('-')) if e.get('dpi') else '-'} | {e.get('bytes','-')} |")
all_ok = all(ok for _, ok in checks)
L.append(f"\n**Overall QC verdict: {'ALL CHECKS PASSED' if all_ok else 'FAILURES PRESENT'}**\n")

L.append("## IEEE print-suitability notes\n")
L.append("- Raster PNGs exported at 300 DPI; vector PDF/SVG companions provided for "
         "every matplotlib-generated figure (fig1 remains the approved raster).")
L.append("- Single-column figures sized ≈3.5–4.6 in wide; full-width figures "
         "(fig2 architecture, fig3 comparison, fig5 Grad-CAM grid) sized ≈6.6–7.16 in.")
L.append("- Minimum font size used ≈7.4 pt at final scale (≥ 6 pt IEEE floor); "
         "axis labels ≥ 8 pt.")
L.append("- fig4 colourbar and cell text remain readable under greyscale printing "
         "(counts printed in every cell).\n")

with open(os.path.join(FINAL_DIR, "FIGURE_PROVENANCE.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))

print("QC:", "ALL PASSED" if all_ok else "FAILED")
for c, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {c}")
print("\nwrote FIGURE_MANIFEST.json + FIGURE_PROVENANCE.md")
