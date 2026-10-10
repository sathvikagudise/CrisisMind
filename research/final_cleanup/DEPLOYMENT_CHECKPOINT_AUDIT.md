# Deployment Checkpoint Audit

Date: 2026-09-01
Scope: Phase 1 + Phase 2 of the "deployment checkpoint parity fix" task.
Verdict: Fusion head is **compatible** to swap, but full logit-parity with the reported
evaluation also requires aligning the feature encoders (see Findings 6-7).

---

## 1. Authoritative final checkpoint (from the reported evaluation)

- Path: `backend/outputs/checkpoints/final_exp/F4_generic_wce_bal_n0.0_s42/best.pth`
- Experiment: `F4_generic_wce_bal_n0.0_s42` (final winner, Gated Fusion v3 / `GatedFusionWide`)
- Reported test metrics (leakage-free): Acc 69.50%, Macro-F1 56.50%, Macro-P 59.58%,
  Macro-R 54.80%, W-F1 67.94%
- Experiment config (from `F4_generic_wce_bal_n0.0_s42/result.json`):
  - lr 1e-4, dropout 0.2, gate_dropout 0.1, best_epoch 13, seed 42
  - img_dim 1280, txt_dim 768, num_classes 3
  - generic (frozen) encoders — see Finding 7
- Full results: `backend/outputs/checkpoints/final_exp/FINAL_TEST_RESULTS.json`

## 2. Currently deployed checkpoint

- Path: `backend/outputs/checkpoints/best_gated_fusion_v3.pth`
- Configured at: `backend/config/shared.py:41` → `GATED_CKPT = CHECKPOINTS_DIR / "best_gated_fusion_v3.pth"`
- Consumed by: `backend/analysis_engine/model_loader.py:44` (`torch.load(GATED_CKPT, map_location=DEVICE)`
  → `gated.load_state_dict(state)`), which only happens inside `load_all_models()`
  (lines 26-70). This is the **single** runtime load point for the fusion head.

## 3. Loading path map

All prediction paths (API, batch processing) go through `AnalysisEngine` (`backend/analysis_engine/engine.py`),
which is constructed exactly once in `backend/api/main.py:80-81` via `load_all_models()`.
- API: `backend/api/main.py:25` imports `load_all_models`; engine created at startup.
- Batch: `backend/batch_processing/processor.py:9` imports `AnalysisEngine` (constructed by caller).
- Fusion head load location: `backend/analysis_engine/model_loader.py:38-47`
  (`GatedFusionWide(img_dim=1280, txt_dim=768, num_classes=3, dropout=FUSION_DROPOUT)`).

No environment-variable overrides exist for checkpoint paths anywhere in the repo.

## 4. Architecture compatibility check

- Deployed class: `backend/models/fusion_models.py:95` `GatedFusionWide(img_dim=1280, txt_dim=768, num_classes=3, dropout=0.3)`
- Final-experiment class: `research/final_experiment/phase4_search.py:40-75` `GatedFusionWide(img_dim=1280, txt_dim=768, num_classes=3, dropout=0.2)`
- **Structurally identical**: gate_net (Dropout(0.1) → Linear(2048→256) → ReLU → Dropout → Linear(256→2) → Sigmoid),
  fc1 (2048→1024), bn1, fc2 (1024→512), bn2, fc3 (512→3).
- Difference: dropout 0.3 (deployed default) vs 0.2 (final). Dropout carries no state and is inert in
  `eval()` mode, so this does not affect inference. No architecture modification needed.

## 5. State-dict compatibility (verified by actual load, Phase 2)

Both `best_gated_fusion_v3.pth` and the F4 `best.pth` are plain `OrderedDict` state dicts with:

- **Identical key set (20 keys)**: gate_net.1.weight/bias, gate_net.4.weight/bias,
  fc1.weight/bias, bn1.(weight,bias,running_mean,running_var,num_batches_tracked),
  fc2.weight/bias, bn2.(...), fc3.weight/bias.
- **Identical shapes**: gate_net.1 (256,2048), gate_net.4 (2,256), fc1 (1024,2048),
  bn1 (1024,), fc2 (1024,512), bn2 (512,), fc3 (512,3). All float32.
- **Same parameter count**: 3,155,719 total elements; 3,152,645 trainable-type
  (matches the 3,152,645 reported in CLEANED_MODEL_REPORT.md).
- **Strict load into deployed-arch `GatedFusionWide` succeeds** for the F4 checkpoint
  (verified: `load_state_dict(F4_best, strict=True)` → no errors).
- Weights naturally differ (the two heads were trained in different experiments).
- `F4_generic_wce_bal_n0.0_s42/best.pth` and its `result.json`, `val_logits.npy`, `val_y.npy`
  were not modified during this audit.

**Conclusion: the F4 head can be swapped in without any architecture or shape changes.**

## 6. Reference inventory for `best_gated_fusion_v3.pth`

| Location | Type |
|---|---|
| `backend/config/shared.py:41` | Definition of `GATED_CKPT` (deployed variant to change) |
| `backend/analysis_engine/model_loader.py:44` | Sole runtime consumer |
| `backend/api/config.py:12` | Re-export (import chain) |
| `backend/api/main.py:26` | Import for startup logging of checkpoint paths (line ~73) |
| `backend/models/efficientnet.py` | Image-classifier builder (unrelated checkpoint) |
| Training-only references (do NOT touch): `backend/training/train_v2.py:641,838`, `backend/training/step2_train.py:476`, `backend/training/train_v3.py:774` (copies winner → `best_gated_fusion_v3.pth`), `backend/training/improved_fusion.py:562,584`, `backend/training/manual_eval_v3.py:79,83` | Historical training scripts |

## 7. Critical parity finding: feature encoders are NOT the evaluated encoders

The reported `F4_generic` evaluation extracted features with **generic frozen encoders**:
- Image: **ImageNet-vanilla** `efficientnet_b0` (torchvision `EfficientNet_B0_Weights.IMAGENET1K_V1`,
  classifier stripped, frozen) — see `research/final_experiment/phase3_extract_features.py`
  arm `generic` (features written to `features_generic.pt`).
- Text: **vanilla** `DistilBertModel.from_pretrained("distilbert-base-uncased")` CLS embedding, frozen.

But `load_all_models()` builds the deployed fusion encoders from task-tuned checkpoints:
- Image encoder ← `best_model.pth` (EfficientNet-B0 backbone, task-tuned; BN `num_batches_tracked=3090`).
  Only 218/358 tensors identical to ImageNet-vanilla; the rest are the task-tuned BN/running stats.
- Text encoder ← `best_text_model.pth` (task-tuned DistilBERT; weights differ from vanilla, ~max 0.0026).

**Consequence:** With only the fusion head swapped, the deployed API computes
`F4_head(best_model_feats, best_text_model_feats)`, whereas the reported evaluation computed
`F4_head(ImageNet_feats, vanillaDBERT_feats)`. The two differ ⇒ deployed logits/predictions will
NOT match the reported 69.50% evaluation for the same inputs.

**For the deterministic parity test (Phase 5) to pass and for the deployed app to genuinely
reproduce the reported leakage-free results, the fusion encoders must also be switched to the
generic frozen encoders** (ImageNet-vanilla B0 + vanilla DistilBERT), matching
`phase3_extract_features.py` arm `generic`. This is still a deployment/loading-path change (no
architecture, fusion-logic, preprocessing, API, training, or dataset changes).

Note: the single-modality classifiers (`image_classifier`, `text_classifier` in `load_all_models`,
items [4/5] and [5/5]) are separate deployed features for image-only / text-only analysis and are
out of scope for this parity fix.

## 8. Preprocessing parity (already aligned)

- Image: `backend/api/preprocessing.py` IMG_TRANSFORM = Resize((224,224)) → ToTensor →
  Normalize(0.485,0.456,0.406 / 0.229,0.224,0.225) — identical to phase3 feature-extraction TF.
- Text: `clean_tweet` (URL/whitespace regex) + `DistilBertTokenizer(truncation, padding, max_length=64)`
  — identical to phase3 (`DistilBertTokenizerFast`, same tokenizer, same max_len 64).

## 9. Files modified (scope decision: full parity, option A)

Decision (confirmed by maintainer): parity fix = **both** the fusion head AND the generic
encoders, per §7 finding.

1. `backend/config/shared.py` — `GATED_CKPT` now → `final_exp/F4_generic_wce_bal_n0.0_s42/best.pth`;
   added `FUSION_ENC_MODE` (= "generic") for fusion encoders.
2. `backend/analysis_engine/model_loader.py` — fusion image/text encoders built from generic
   frozen encoders when parity mode is selected; `image_clf`/`text_clf` unchanged.
3. `backend/models/fusion_models.py` — `ImageFeatureExtractor` / `TextFeatureExtractor` accept
   `checkpoint_path=None` → generic pretrained (ImageNet-vanilla / vanilla DistilBERT);
   loading-path only, forward logic unchanged.

No other files affected. No retraining, no dataset/test-label changes, no
`FINAL_TEST_RESULTS.json` / `FINAL_REPORT.md` / paper / PPT edits.

## 10. Outcome of the decision

Option A (head + encoders) was selected. Applied, backend restarted, `/health` OK, real
inference + Grad-CAM verified, and a 12-sample deterministic parity test (direct F4 path vs
deployed path) matched within FP tolerance (max class dev 0, max prob dev 7.7e-07, max conf dev
7.7e-05). See `DEPLOYMENT_CHECKPOINT_FIX_REPORT.md`.