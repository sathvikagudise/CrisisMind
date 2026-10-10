# CrisisMind — Model Validation Report

**Date:** July 28, 2026
**Models Validated:** EfficientNet-B0 (image), DistilBERT (text), GatedFusion (multimodal)

---

## Model Inventory

| Model | Checkpoint | Size | Parameters | Architecture |
|-------|-----------|------|------------|--------------|
| Image Encoder | best_model.pth | 16.3 MB | ~5.3M | EfficientNet-B0 backbone (features + avgpool) |
| Image Classifier | best_model.pth | 16.3 MB | ~5.3M | EfficientNet-B0 + Dropout(0.2) + Linear(1280→3) |
| Text Encoder | best_text_model.pth | 267.9 MB | ~66M | DistilBERT (6 layers, 768 hidden) |
| Text Classifier | best_text_model.pth | 267.9 MB | ~66M | DistilBERT + PreClassifier(768→768) + Classifier(768→3) |
| Gated Fusion | best_gated_fusion.pth | 4.7 MB | ~1.3M | Gate(2048→64→2) + FC(2048→512→3) |
| Gated Fusion v2 | best_gated_fusion_v2.pth | 5.0 MB | ~1.3M | Same architecture, different weights |

---

## Checkpoint Load Verification

| Checkpoint | strict=True | Missing Keys | Unexpected Keys | Status |
|-----------|-------------|--------------|-----------------|--------|
| best_gated_fusion.pth | ✅ Yes | None | None | ✅ VALID |
| best_model.pth (as encoder) | ⚠️ No | None (classifier stripped) | num_batches_tracked | ✅ OK |
| best_model.pth (as classifier) | ⚠️ No | None | None | ✅ OK |
| best_text_model.pth (as encoder) | ⚠️ No | None (classifier stripped) | pre_classifier, classifier | ✅ OK |
| best_text_model.pth (as classifier) | ⚠️ No | None | None | ✅ OK |

---

## Performance Summary (Test Set: 529 samples)

| Model | Accuracy | F1 Macro | F1 Weighted | Recall Little | Recall Mild | Recall Severe |
|-------|----------|----------|-------------|---------------|-------------|---------------|
| Gated Fusion | 63.52% | 53.75% | 64.15% | 43.7% | 44.4% | 75.0% |
| Image Only | 58.41% | 48.69% | 59.59% | 47.9% | 34.1% | 69.9% |
| Text Only | 56.52% | 47.48% | 57.26% | 67.6% | 24.6% | 66.3% |

### Baseline Comparison
- **Random (3-class):** 33.3% accuracy
- **Majority class (always Severe):** 62.7% accuracy
- **Gated Fusion:** 63.5% accuracy (+0.8% above majority baseline)

---

## Gate Analysis

| Metric | Image Gate | Text Gate |
|--------|-----------|-----------|
| Mean | 0.173 | 0.014 |
| Min | 0.107 | 0.004 |
| Max | 0.261 | 0.031 |

**Interpretation:** Both gates are well below 0.5, indicating the fusion layer has learned to suppress both input signals. The text gate is essentially zero, meaning the multimodal model operates nearly as an image-only model with a severe-class bias.

---

## Prediction Distribution

| Model | Little | Mild | Severe | Severe % |
|-------|--------|------|--------|----------|
| Ground Truth | 71 | 126 | 332 | 62.8% |
| Gated Fusion | 78 | 138 | 313 | 59.2% |
| Image Only | 109 | 123 | 297 | 56.1% |
| Text Only | 161 | 80 | 288 | 54.4% |

---

## Key Findings

1. **Label mapping is correct** across all stages (training → model → API → frontend)
2. **Preprocessing is consistent** between training and inference (ImageNet normalization, DistilBERT tokenizer)
3. **All checkpoints load successfully** with expected key filtering
4. **Gated Fusion v2 exists but is NOT used** — API loads v1
5. **Class imbalance** (62.7% severe) is the primary driver of severe bias
6. **Gate collapse** means the fusion model effectively ignores both modalities

---

## Recommendations

1. **Immediate:** Switch API to load `best_gated_fusion_v2.pth` (minor improvement expected)
2. **Short-term:** Add confidence thresholding — reject predictions below 55% confidence
3. **Long-term:** Retrain with proper class balancing, gate regularization, and diverse text data
