# Text Baseline Model — Evaluation Summary

**Model:** DistilBERT (distilbert-base-uncased)

## Dataset

- Total unique text samples: 3198
- Train: 2238 | Validation: 480 | Test: 480

### Class Distribution

| Label | Count |
|---|---|
| little_or_no_damage | 416 |
| mild_damage | 739 |
| severe_damage | 2043 |

## Test Set Performance

| Metric | Value |
|---|---|
| Accuracy | 0.5250 |
| Precision (weighted) | 0.5741 |
| Recall (weighted) | 0.5250 |
| F1 Score (weighted) | 0.5328 |
| Precision (macro) | 0.4291 |
| Recall (macro) | 0.4817 |
| F1 Score (macro) | 0.4221 |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| little_or_no_damage | 0.2614 | 0.6452 | 0.3721 | 62 |
| mild_damage | 0.2836 | 0.1712 | 0.2135 | 111 |
| severe_damage | 0.7423 | 0.6287 | 0.6808 | 307 |

## Visualizations

![Confusion Matrix](../outputs/plots/text_confusion_matrix.png)

