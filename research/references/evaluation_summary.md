# Image Baseline Model � Evaluation Summary

**Model:** EfficientNet-B0 (pretrained on ImageNet)

## Dataset

- Total damage samples: 3526
- Train: 2468 | Validation: 529 | Test: 529

### Class Distribution

| Label | Count |
|---|---|
| little_or_no_damage | 475 |
| mild_damage | 839 |
| severe_damage | 2212 |

## Test Set Performance

| Metric | Value |
|---|---|
| Accuracy | 0.5841 |
| Precision (weighted) | 0.6154 |
| Recall (weighted) | 0.5841 |
| F1 Score (weighted) | 0.5959 |
| Precision (macro) | 0.4809 |
| Recall (macro) | 0.5063 |
| F1 Score (macro) | 0.4869 |

## Visualization

![Confusion Matrix](../../backend/outputs/plots/confusion_matrix.png)

![Training Curves](../../backend/outputs/plots/training_curves.png)

![Class Distribution](../../backend/outputs/plots/class_distribution.png)

![Split Distribution](../../backend/outputs/plots/split_distribution.png)

