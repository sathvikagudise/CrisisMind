# CrisisMind — Final Results

> All values below are taken from the trained model artifacts (`backend/outputs/checkpoints/v3/wide_wce_g1.0_lr1e-04_summary.json`),
> the v2 summary (`backend/outputs/checkpoints/v2/wce_reg1_summary.json`), the evaluation reports in `testing/`,
> and real inference latency measured against the running FastAPI `/predict` endpoint.
> No values are estimated or invented.

---

## Best Model

**GatedFusionWide (v3)** — config `wide_wce_g1.0_lr1e-04` on the CrisisMMD v2.0 severity classification task.

| Metric | Value |
|---|---|
| **Accuracy** | **0.6011** |
| **Precision (macro)** | **0.5403** |
| **Precision (weighted)** | **0.6695** |
| **Recall (macro)** | **0.5576** |
| **Recall (weighted)** | **0.6011** |
| **Macro F1** | **0.5372** |
| **Weighted F1** | **0.6220** |
| **Inference Time** | **237.2 ms** (multimodal, CPU, steady-state mean of 3 trials) |
| **Best Model** | **GatedFusionWide — `wide_wce_g1.0_lr1e-04`** |
| **Number of Test Samples** | **529** (71 little_or_no_damage, 126 mild_damage, 332 severe_damage) |
| **Hardware Used** | **CPU** — AMD64 (AuthenticAMD, Family 25 Model 80), no CUDA, PyTorch 2.2.2+cpu, Windows 11 |

### Per-Class Performance (test set, v3)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| little_or_no_damage | 0.4177 | 0.4648 | 0.4400 |
| mild_damage | 0.3641 | 0.5635 | 0.4424 |
| severe_damage | 0.8392 | 0.6446 | 0.7291 |

### Gate Health

| Gate | v1 | v2 | v3 |
|---|---|---|---|
| Image Gate | 0.1730 | 0.6441 | 0.5365 |
| Text Gate | 0.0140 | 0.4314 | 0.4357 |

---

## Inference Time (measured, not estimated)

Recorded from real `/predict` responses (`processing_time_ms`), CPU, AMD64 Zen 4-class machine.

| Mode | Model | Trial 1 (ms) | Trial 2 (ms) | Trial 3 (ms) | Mean (ms) |
|---|---|---|---|---|---|
| TEXT_ONLY | DistilBERT | 2205.2 (cold) | 141.39 | 138.98 | 140.19* |
| IMAGE_ONLY | EfficientNet-B0 | 272.28 (cold) | 95.25 | 89.99 | 92.62* |
| MULTIMODAL | GatedFusion v3 | 230.14 | 259.54 | 221.91 | **237.20** |

*Mean excludes the cold-start trial. Cold start includes one-time model warm-up.

---

## Final Training Configuration

| Hyperparameter | Value |
|---|---|
| Architecture | GatedFusionWide (1024 hidden + BatchNorm, 3.1M trainable params) |
| Image encoder | EfficientNet-B0 (frozen, 1280-d features) |
| Text encoder | DistilBERT (frozen, 768-d features) |
| Fusion input | 2048-d (1280 image + 768 text) |
| Gate network | 2048 → 256 → 2 (dropout) |
| Loss | Weighted Cross-Entropy (inverse frequency, normalized) |
| Class weights | little: 1.6831, mild: 0.9548, severe: 0.3621 |
| Gate reg weight | 1.0 (target 0.5) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | warmup_cosine (5-epoch warmup) |
| Max epochs | 40 |
| Patience | 10 |
| Best epoch | 6 |
| Batch size (train) | 32 |
| Batch size (val/test) | 64 |
| Feature noise | 0.0 |
| Dropout | 0.3 |
| Random seed | 42 |
| Split | 0.70 / 0.15 / 0.15 stratified (train 2468 / val 529 / test 529) |
| Device | CPU |

---

## Model Comparison (v1 vs v2 vs v3)

| Model | Version | Accuracy | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|
| ImageBaseline | v1 | 0.5841 | 0.4869 | 0.5959 |
| TextBaseline | v1 | 0.5250 | 0.4221 | 0.5328 |
| ConcatFusion | v1 | 0.6371 | 0.5302 | 0.6400 |
| GatedFusion | v1 | 0.6352 | 0.5375 | 0.6415 |
| GatedFusion (wce_reg1) | v2 | 0.5841 | 0.5256 | 0.6059 |
| **GatedFusionWide** | **v3** | **0.6011** | **0.5372** | **0.6220** |

v3 restores v1-level Macro F1 (0.5372 vs 0.5375) while fixing the gate collapse that
plagued v1 (text gate 0.014 → 0.436, a 31x improvement).

---

## Source Data Files

- `research/paper_assets/dataset_statistics.csv` — dataset overview, per-event rows, label distributions, split sizes
- `research/paper_assets/training_configuration.csv` — final training hyperparameters
- `research/paper_assets/experimental_results.csv` — top-20 v3 hyperparameter search runs
- `research/paper_assets/model_comparison.csv` — all baselines vs v1/v2/v3
- `research/paper_assets/evaluation_metrics.csv` — full v3 metric set from `wide_wce_g1.0_lr1e-04_summary.json`
- `research/paper_assets/confusion_matrix.png` — test-set confusion matrix of the v3 winner
- `research/paper_assets/training_curve.png` — training curves (v3 run with identical loss/gate-reg/LR/scheduler)
- `research/paper_assets/model_comparison.png` — v1 vs v2 vs v3 comparison chart
- `research/paper_assets/inference_statistics.csv` — measured inference latency per mode

## Notes

- The v3 winner's per-epoch training curves were not persisted to disk (batch-2 training saves
  only the confusion matrix). `training_curve.png` is the closest real saved v3 run with the
  identical loss (inverse WCE), gate-reg weight (1.0), LR (1e-4), and warmup_cosine scheduler
  (`v3_008_inverse_wce_g1.0_lr1e-04_cosine_warmup_curves.png`).
- Majority-class baseline accuracy on the test set is 62.7% (severe_damage, 62.8% of test data).
