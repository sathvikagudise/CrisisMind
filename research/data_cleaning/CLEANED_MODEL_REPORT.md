# Cleaned-Dataset Model Report — Gated Fusion v3

**Phase 6 deliverable.** Results of retraining the published Gated Fusion v3
(`GatedFusionWide`, config `wide_wce_g1.0_lr1e-04`) on the **cleaned** CrisisMMD severity
subset (3,105 samples) with the **leak-free** split from Phase 4.

## Training configuration (identical to the original v3 winner)

| Hyperparameter | Value |
|---|---|
| Architecture | GatedFusionWide (1024 hidden + BatchNorm, 3,152,645 trainable params) |
| Image encoder | EfficientNet-B0 (frozen, 1280-d) — unchanged checkpoint `best_model.pth` |
| Text encoder | DistilBERT (frozen, 768-d) — unchanged checkpoint `best_text_model.pth` |
| Loss | Weighted Cross-Entropy (inverse frequency, normalized) |
| Class weights | little: 1.6805, mild: 0.9832, severe: 0.3363 |
| Gate reg weight | 1.0 (target 0.5) |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| Scheduler | warmup_cosine (5-epoch warmup) |
| Max epochs / patience | 40 / 10 |
| Batch (train / val/test) | 32 / 64 |
| Sampler | WeightedRandomSampler (inverse), replacement=True |
| Feature noise | 0.0 |
| Random seed | 42 |
| Population / split | 3,105 samples; leak-free stratified 70/15/15 → 2,173 / 467 / 465 |
| Hardware | CPU (AMD64, no CUDA) |
| Wall time | 0.35 min (21 s) |

The recipe is byte-for-byte the same as `wide_wce_g1.0_lr1e-04` in
`backend/training/train_v3_batch2.py` (same model class, same loss, same optimizer,
scheduler, sampler, gate reg, seeds). Early stopping fired at epoch 7 with best epoch 6 —
**identical** to the original run (best epoch 6), confirming reproducible behavior.

## Test results (465 leak-free test samples)

| Metric | Value |
|---|---|
| **Accuracy** | **0.5699** |
| **Precision (macro)** | **0.5007** |
| **Precision (weighted)** | **0.6778** |
| **Recall (macro)** | **0.5134** |
| **Recall (weighted)** | **0.5699** |
| **Macro F1** | **0.4829** |
| **Weighted F1** | **0.5972** |
| **Image Gate** | **0.5579** |
| **Text Gate** | **0.4177** |

### Per-class performance (cleaned test set, n=465)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| little_or_no_damage | 0.2951 | 0.3000 | 0.2975 |
| mild_damage | 0.3367 | 0.6408 | 0.4415 |
| severe_damage | 0.8702 | 0.5993 | 0.7098 |

### Confusion matrix (rows=actual, cols=predicted: little / mild / severe)

```
[[ 18  34   8]
 [ 18  66  19]
 [ 25  96 181]]
```

## Inference latency (CPU, steady-state)

| Component | ms | Source |
|---|---|---|
| Image encoder (EfficientNet-B0) | 92.62 | unchanged — carried from FINAL_RESULTS.md |
| Text encoder (DistilBERT) | 140.19 | unchanged — carried from FINAL_RESULTS.md |
| Fusion head (GatedFusionWide) | 6.98 | measured here (batch 64, mean of 3 trials: 8.96/6.63/5.34 ms) |
| **Total pipeline (est.)** | **~239.8 ms** | same encoder cost as original (237.2 ms) |

Encoders are the *same frozen checkpoints* as the original model, so per-image/per-text
cost is unchanged; only fusion-head weights differ (same architecture → same FLOPs).

## Artifacts

| Artifact | Location |
|---|---|
| Best weights | `backend/outputs/checkpoints/v3_clean/cleaned_v3_wide_wce_g1.0_lr1e-04_best.pth` |
| Training summary (JSON) | `backend/outputs/checkpoints/v3_clean/cleaned_v3_wide_wce_g1.0_lr1e-04_summary.json` |
| Eval summary (JSON) | `backend/outputs/checkpoints/v3_clean/cleaned_v3_wide_wce_g1.0_lr1e-04_eval.json` |
| Confusion matrix (PNG) | `backend/outputs/checkpoints/v3_clean/cleaned_v3_wide_wce_g1.0_lr1e-04_cm.png` |
| Predictions (npz) | `backend/outputs/checkpoints/v3_clean/cleaned_v3_wide_wce_g1.0_lr1e-04_preds.npz` |
| Metrics CSV | `research/paper_assets/cleaned_dataset_results.csv` |
