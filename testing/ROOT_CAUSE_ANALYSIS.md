# CrisisMind — Severe Damage Bias Root Cause Analysis

**Date:** July 28, 2026
**Issue:** All predictions return "Severe Damage" regardless of input
**Status:** ROOT CAUSE IDENTIFIED — Multiple contributing factors

---

## Executive Summary

The CrisisMind models do **not** literally predict "Severe Damage" for every input. On the formal test set (529 samples), the multimodal model achieves 63.5% accuracy and 53.75% macro F1. However, the system exhibits a **strong systematic bias toward the "Severe Damage" class** due to:

1. **Extreme class imbalance** in training data (62.7% severe)
2. **Gated fusion suppresses both modalities** (near-zero gate values)
3. **Out-of-distribution inputs** from users (clean English vs tweet-style training data)

These factors combine to produce low-confidence predictions (40–60%) that default to the majority class, especially for out-of-distribution inputs.

---

## Phase 1: Label Mapping Verification ✅

**Result: CORRECT — No bug in label mapping**

| Stage | Label 0 | Label 1 | Label 2 |
|-------|---------|---------|---------|
| Training LABEL_MAP | little_or_no_damage | mild_damage | severe_damage |
| API SEVERITY_MAP | Little or No Damage | Mild Damage | Severe Damage |
| Frontend display | "Little or No Damage" | "Mild Damage" | "Severe Damage" |

All three stages map consistently: 0 → Little, 1 → Mild, 2 → Severe.

---

## Phase 2: Checkpoint Verification ✅

**Result: CORRECT — All checkpoints load successfully**

| Checkpoint | Size | Loaded By | Status |
|-----------|------|-----------|--------|
| best_model.pth (EfficientNet) | 16.3 MB | img_encoder + image_classifier | ✅ Loads with strict=False |
| best_text_model.pth (DistilBERT) | 267.9 MB | txt_encoder + text_classifier | ✅ Loads with strict=False |
| best_gated_fusion.pth (v1) | 4.7 MB | gated_model | ✅ Loads with strict=True |

**Note:** `best_gated_fusion_v2.pth` (5.0 MB) exists from improved training but is **NOT loaded** by the API. The API loads v1.

---

## Phase 3: Preprocessing Verification ✅

**Result: CORRECT — No preprocessing mismatch**

| Stage | Resize | Normalize | Tokenizer |
|-------|--------|-----------|-----------|
| Training (improved_fusion.py) | 224×224 | ImageNet [0.485,0.456,0.406]/[0.229,0.224,0.225] | DistilBERT, max_len=64 |
| Inference (preprocessing.py) | 224×224 | ImageNet [0.485,0.456,0.406]/[0.229,0.224,0.225] | DistilBERT, max_len=64 |

---

## Phase 4: Training Data Distribution ⚠️

**Result: EXTREME CLASS IMBALANCE — Primary root cause**

```
Training data distribution (3,526 total samples):
  little_or_no_damage   475  (13.5%)
  mild_damage           839  (23.8%)
  severe_damage       2,212  (62.7%)  ← DOMINANT CLASS
```

A trivial "always predict severe" baseline would achieve **62.7% accuracy**. The multimodal model's 63.5% accuracy is only 0.8 percentage points above this baseline.

**Impact:** The model learns that predicting "Severe" is the lowest-risk strategy, especially when uncertain. This creates a prior toward the majority class.

---

## Phase 5: Full Test Set Evaluation

**Result: Models are functional but mediocre**

### Multimodal (Gated Fusion) — What the API uses
```
Accuracy:     63.52%
F1 (macro):   53.75%
F1 (weighted): 64.15%

Per-class recall:
  Little or No Damage:  43.7%  (31/71)
  Mild Damage:          44.4%  (56/126)
  Severe Damage:        75.0%  (249/332)

Prediction distribution:
  Little:  78  (14.7%)
  Mild:   138  (26.1%)
  Severe: 313  (59.2%)  ← close to 62.7% training ratio
```

### Image-Only (EfficientNet-B0)
```
Accuracy:     58.41%
F1 (macro):   48.69%
Per-class recall: Little=47.9%, Mild=34.1%, Severe=69.9%
```

### Text-Only (DistilBERT)
```
Accuracy:     56.52%
F1 (macro):   47.48%
Per-class recall: Little=67.6%, Mild=24.6%, Severe=66.3%
```

---

## Phase 6: Gated Fusion Gate Analysis ⚠️

**Result: Both gates near zero — model suppresses input signals**

From manual test inference:
```
Image gate: 0.107 – 0.261  (mean ~0.17)
Text gate:  0.004 – 0.031  (mean ~0.014)
```

**What this means:**
```python
gated_img = g_img * img_feat    # img_feat * 0.17 ≈ 17% of original
gated_txt = g_txt * txt_feat    # txt_feat * 0.014 ≈ 1.4% of original (NEARLY ZERO)
fused = [gated_img, gated_txt]  # 2048-d vector, mostly from image
fc_output = FC(fused)           # FC layers dominate, biased toward severe
```

The gated fusion has learned to **nearly eliminate the text signal** and **heavily dampen the image signal**. The output is then dominated by the bias terms in the final FC layer, which default to the majority class.

---

## Phase 7: Manual Test (Out-of-Distribution) Results

**Result: Catastrophic failure on clean English text**

```
Text-only on 15 manual samples:
  Little or No Damage:  0/5 correct  (0%)
  Mild Damage:          0/5 correct  (0%)
  Severe Damage:        5/5 correct  (100%)
  Overall:              5/15 correct  (33.3%)

Image-only on 15 manual test images:
  Distribution: Little=5, Mild=2, Severe=8 (not all severe)

Multimodal on 5 manual test images:
  Distribution: Little=1, Mild=0, Severe=4
  All gates near zero: img≈0.15, txt≈0.01
```

**Why manual tests fail:**
- Training data = cleaned tweets (short, informal, disaster hashtags)
- Manual tests = clean English sentences (formal, descriptive)
- The DistilBERT model was fine-tuned on tweet distributions and does not generalize to clean English
- The text gate is near zero, so the text model's poor OOD performance doesn't directly hurt multimodal — but it means the fusion is effectively image-only with a Severe bias baked in

---

## Root Cause Summary

| # | Cause | Severity | Fixable? |
|---|-------|----------|----------|
| 1 | **Class imbalance** (62.7% severe in training data) | HIGH | Retrain with oversampling/SMOTE |
| 2 | **Gated fusion suppresses both modalities** (near-zero gates) | HIGH | Retrain with gate regularization or different architecture |
| 3 | **OOD inputs** (clean English vs tweet-style training data) | MEDIUM | Add diverse text samples to training |
| 4 | **Text model barely trained** (pre_classifier/classifier weights near random init) | MEDIUM | Retrain with higher LR for classifier head |
| 5 | **API loads v1 checkpoint, not v2** | LOW | Update GATED_CKPT to point to v2 |
| 6 | **Low overall model capacity** (53.75% F1 macro) | HIGH | Larger model, more data, better architecture |

---

## Recommended Fixes

### Quick Fix (no retraining)
1. **Update GATED_CKPT** to point to `best_gated_fusion_v2.pth`
2. **Add confidence threshold** — if confidence < 60%, return "Insufficient data for reliable classification"
3. **Add input validation** — warn users that text must be disaster-related

### Proper Fix (retraining required)
1. **Fix class imbalance**: Use SMOTE oversampling or aggressive class weights
2. **Retrain gated fusion** with gate regularization to prevent gates from collapsing to zero
3. **Train text classifier** with higher learning rate for classifier head (currently the head barely learns)
4. **Add diverse training data** including clean English descriptions alongside tweets
5. **Use focal loss with stronger alpha** for minority classes

---

## Files Generated

| File | Description |
|------|-------------|
| `testing/INFERENCE_DEBUG.csv` | Per-sample raw logits for all 3 models |
| `testing/ROOT_CAUSE_ANALYSIS.md` | This document |
| `testing/outputs/plots/confusion_matrix_*.png` | Confusion matrices for all 3 modes |

---

## Conclusion

The "all Severe Damage" behavior is **not a code bug** — it is an expected consequence of:
- Training data dominated by severe damage (62.7%)
- A gated fusion mechanism that learned to suppress input signals
- Out-of-distribution inputs from users

The models ARE functional (63.5% accuracy > 33.3% random), but their strong majority-class bias makes them unreliable for real-world use, especially on clean English text inputs that differ from the tweet-style training data.
