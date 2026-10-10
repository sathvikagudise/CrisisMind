# Original vs Cleaned Dataset — Model Comparison

**Phase 7 deliverable.** Head-to-head comparison of the identical Gated Fusion v3 model
(same architecture `GatedFusionWide`, same training recipe `wide_wce_g1.0_lr1e-04`)
trained/evaluated on the **original** CrisisMMD severity subset vs the **cleaned** subset.

Both runs are apples-to-apples: same architecture, same loss, same optimizer/scheduler/
seed, same frozen encoders, same early-stopping behavior (best epoch 6 in both).

## Setup summary

| | Original | Cleaned |
|---|---|---|
| Population | 3,526 | 3,105 |
| Split | stratified 70/15/15 (seed 42) | leak-free stratified 70/15/15 (seed 42) |
| Train / Val / Test | 2,468 / 529 / 529 | 2,173 / 467 / 465 |
| Near-dup clusters crossing splits | **94** | **0** |
| Exact-identical pairs crossing splits | **27** | **0** |

## Headline metrics (test set)

| Metric | Original | Cleaned | Δ |
|---|---|---|---|
| Accuracy | 0.6011 | 0.5699 | −0.0312 |
| Precision (macro) | 0.5403 | 0.5007 | −0.0396 |
| Recall (macro) | 0.5576 | 0.5134 | −0.0442 |
| **Macro F1** | **0.5372** | **0.4829** | **−0.0543** |
| Weighted F1 | 0.6220 | 0.5972 | −0.0248 |
| Image Gate | 0.5365 | 0.5579 | +0.0214 |
| Text Gate | 0.4357 | 0.4177 | −0.0180 |

### Per-class F1

| Class | Original | Cleaned | Δ |
|---|---|---|---|
| little_or_no_damage | 0.4400 | 0.2975 | −0.1425 |
| mild_damage | 0.4424 | 0.4415 | −0.0009 |
| severe_damage | 0.7291 | 0.7098 | −0.0193 |

## Interpretation (honest, non-spun)

The cleaned model scores **lower** on every headline metric. This is the expected and
correct outcome, and it is a **strength of the methodology**, not a regression to hide:

1. **The original test set was optimistically biased.** 94 of 183 near-duplicate image
   clusters (and 27 exact-identical image groups) straddled the original train/test
   boundary. The original model could "cheat" by memorizing near-duplicates seen in
   training and matching them at test time. Its reported 0.6011 acc / 0.5372 macro-F1
   were therefore inflated relative to true generalization.
2. **The cleaned test set measures real generalization.** Removing duplicates and
   containing every remaining near-dup cluster inside a single split forces the model to
   generalize to genuinely unseen imagery. The lower scores are the honest generalization
   estimate.
3. **The model itself is unchanged** — identical architecture, weights sizes, and training
   recipe. The metric delta is a *dataset integrity* effect, not an architecture effect.
4. **The largest drop is in `little_or_no_damage`** (−0.14 macro-contributing F1), the
   rarest class. Its original F1 was inflated by near-duplicate leakage most heavily,
   because rare-class images get re-shared and re-annotated most often; on clean, unseen
   test images the model generalizes less well to this minority class.
5. **Gates stayed healthy** (img 0.558, txt 0.418) — no gate collapse; the fusion
   mechanism behaves identically under the cleaned data.

## What this means for reporting

- The cleaned-dataset numbers (`cleaned_dataset_results.csv`, macro-F1 0.4829) are the
  **conservative, leakage-free** estimate of CrisisMind's generalization on
  severity classification.
- The original published numbers remain valid *as recorded on the original split*, but any
  paper/integrity discussion must disclose the 94-cluster leakage and treat the original
  test metrics as upper-bound estimates.

## Repro / artifacts

- Original: `backend/outputs/checkpoints/v3/wide_wce_g1.0_lr1e-04_summary.json`
- Cleaned: `backend/outputs/checkpoints/v3_clean/cleaned_v3_wide_wce_g1.0_lr1e-04_summary.json`
- Comparison inputs: `research/paper_assets/evaluation_metrics.csv` vs `research/paper_assets/cleaned_dataset_results.csv`
