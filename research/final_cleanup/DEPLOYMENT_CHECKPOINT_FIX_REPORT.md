# Deployment Checkpoint Fix Report

Date: 2026-09-01
Status: COMPLETE — deployed app now loads the authoritative final checkpoint and
reproduces the reported evaluation path within floating-point tolerance.

---

## 1. Old (previously deployed) checkpoint

- Path: `backend/outputs/checkpoints/best_gated_fusion_v3.pth`
- Role: Gated Fusion v3 (`GatedFusionWide`) head served by the live API before this fix.
- Status after fix: **retained on disk, unmodified** (SHA-256 preserved) as historical evidence.
- Configured previously at `backend/config/shared.py:41` (`GATED_CKPT`).

## 2. New (authoritative final) checkpoint

- Path: `backend/outputs/checkpoints/final_exp/F4_generic_wce_bal_n0.0_s42/best.pth`
- Experiment: `F4_generic_wce_bal_n0.0_s42` — the **final selected leakage-free model**
  (reported test: Acc 69.50%, Macro-F1 56.50%, W-F1 67.94%, CM [[24,19,18],[11,37,51],[8,26,242]]).
- Config (from `result.json`): lr 1e-4, dropout 0.2, gate_dropout 0.1, WCE + balanced sampling,
  noise 0.0, softmax head, best_epoch 13, seed 42.
- Status after fix: **unmodified**; only referenced by path (no copy/rename created).
- Full official results: `backend/outputs/checkpoints/final_exp/FINAL_TEST_RESULTS.json`.

## 3. Files changed (exactly three)

| File | Change |
|---|---|
| `backend/config/shared.py` | `GATED_CKPT` now → `final_exp/F4_generic_wce_bal_n0.0_s42/best.pth`; added `FUSION_ENC_MODE` (="generic") deployment-parity flag + explanatory comment. |
| `backend/models/fusion_models.py` | `ImageFeatureExtractor` / `TextFeatureExtractor` now accept `checkpoint_path=None` → load **generic** pretrained encoders (ImageNet-vanilla EfficientNet-B0 / vanilla DistilBERT). Forward logic, architecture, and dims unchanged. |
| `backend/analysis_engine/model_loader.py` | Pass `None` encoder checkpoints when `FUSION_ENC_MODE=="generic"` so the fusion encoders match the encoders used in the reported evaluation; log lines updated; loading path otherwise identical. |

No changes to: model architecture, fusion/gate logic, training scripts, dataset, test labels,
preprocessing, API design, frontend, `FINAL_TEST_RESULTS.json`, `FINAL_REPORT.md`, figures, or PPT.

## 4. Compatibility verification (Phase 2)

- Both checkpoints are plain `OrderedDict` state dicts with **identical 20-key set** and matching
  shapes (gate_net 256/2 heads, fc1 2048→1024, bn1 1024, fc2 1024→512, bn2 512, fc3 512→3).
- Parameter counts identical: 3,155,719 total elements / 3,152,645 trainable-type
  (matches CLEANED_MODEL_REPORT.md).
- `GatedFusionWide(1280, 768, 3)`. `load_state_dict(F4_best.pth, strict=True)` **succeeds**.
  No architecture change required. Dropout mismatch (deployed default 0.3 vs final 0.2) is inert
  in `eval()` mode.

## 5. App loads successfully (Phase 4)

- Backend restarted: `uvicorn backend.api.main:app --host 127.0.0.1 --port 8000`
- Startup log confirms:
  - `GATED model → ...\final_exp\F4_generic_wce_bal_n0.0_s42\best.pth  [exists=True]`
  - `[3/5] Loading GatedFusionWide classifier (v3)...` → `all 5 models loaded successfully`
  - `Application startup complete`
- `/health` → `{"status":"healthy","models_loaded":5}`
- Live inference (multimodal, image-only, text-only) and Grad-CAM all return valid predictions.

## 6. Direct vs API prediction parity (Phase 5) — PASS

Deterministic test (`research/final_cleanup/checkpoint_parity_test.py`, output
`checkpoint_parity.json`), 12 test-split samples compared two pipelines:
- **A = exact evaluation path**: F4 `best.pth` head applied to cached **generic** features
  (`features_generic.pt`), i.e. precisely the inputs of the reported evaluation.
- **B = deployed path**: `backend/api/preprocessing` → generic ImageNet-vanilla B0 +
  vanilla DistilBERT encoders → deployed `GatedFusionWide` (F4 head) → softmax → argmax.

Result (max across 12 samples):

| Quantity | Max deviation | Tolerance |
|---|---|---|
| predicted class | 0 | 0 |
| class probabilities | 7.7e-07 | 1e-6 |
| confidence (%) | 7.7e-05 | 1e-4 |
| fused logits | 5e-07 | — |
| gate values | 0.0 | — |
| image features | 2e-06 | — |
| text features | 0.0 | — |

**Conclusion: deployed app == reported evaluation path within floating-point tolerance.**
This also means the app now reproduces the leakage-free F4_generic metrics on identical inputs.

## 7. Deployment latency (Phase 6) — reported SEPARATELY

Full deployed HTTP stack (CPU), warm, n=12 per mode; including multipart parse + preprocessing
+ encoders + fusion head + response builder. `research/final_cleanup/deployed_latency.json`:

| Mode | mean ms | std ms | p50 | p95 |
|---|---|---|---|---|
| MULTIMODAL | 171.6 | 38.0 | 151.9 | 228.7 |
| IMAGE_ONLY | 76.3 | 8.0 | 73.2 | 90.0 |
| TEXT_ONLY | 105.5 | 11.6 | 103.9 | 122.2 |

NOT comparable directly to the final-experiment reported
`133.38 ± 17.67 ms` (pipeline) / `1.349 ± 0.172 ms` (head): different harness, environment
(this is the full API stack over HTTP). Reported here separately per task instructions.

## 8. Checkpoint integrity

- `best_gated_fusion_v3.pth` — retained, unmodified (historical evidence).
- `F4_generic_wce_bal_n0.0_s42/best.pth` — unmodified (referenced by path only).
- `FINAL_TEST_RESULTS.json`, `FINAL_REPORT.md`, paper/PPT — untouched.
- Modified source files (SHA-256 prefix): shared.py `655c6d6e…`, fusion_models.py `9426ffcb…`,
  model_loader.py `f159d3b7…`.

## 9. Detail: why the encoders had to change (parity rationale)

The reported evaluation (arm `generic` in `research/final_experiment/phase3_extract_features.py`)
extracted features with **frozen generic** encoders — ImageNet-pretrained torchvision
EfficientNet-B0 and vanilla `distilbert-base-uncased`. The previously-deployed API, however, fed
the fusion head features from **task-tuned** `best_model.pth`/`best_text_model.pth`. Those produce
different feature vectors (e.g. BN `num_batches_tracked=3090` on `best_model.pth`), so swapping
only the head would NOT have reproduced the reported metrics. Aligning encoders to the
experiment's generic frozen encoders (via `FUSION_ENC_MODE="generic"`) is what makes parity real.
Single-modality classifiers (image-only / text-only) were intentionally left task-tuned, as those
are separate deployed features outside this fusion-checkpoint parity fix.

## 10. Files & evidence produced

- `research/final_cleanup/DEPLOYMENT_CHECKPOINT_AUDIT.md` — Phase 1+2 audit
- `research/final_cleanup/checkpoint_parity_test.py` + `checkpoint_parity.json` — Phase 5
- `research/final_cleanup/latency_measure.py` + `deployed_latency.json` — Phase 6
- this report — Phase 7

## 11. Remaining mismatches / caveats

- **None functionally.** Model-name string still reads `GatedFusion v3` in API responses
  (architecture is the same v3 trunk; label unchanged intentionally to avoid API-design changes).
- Image-only / text-only modes remain on task-tuned single-modality classifiers (out of scope;
  documented in §9).
- Deployed latency higher than the final-experiment head/feature-extraction timings — different
  harnesses, reported separately (§7); not a correctness issue.
- Deployment currently on CPU at `127.0.0.1:8000` (demo env), consistent with pre-fix setup.

---

**FINAL EVALUATION CHECKPOINT == DEPLOYED CHECKPOINT** — verified by identical state-dict
compatibility and a deterministic 12-sample parity test (class, probabilities, confidence, fused
outputs) inside floating-point tolerance; the deployed app loads the authoritative
`F4_generic_wce_bal_n0.0_s42/best.pth` with generic frozen encoders and reproduces the reported
leakage-free evaluation path.