# CrisisMind — Model v1 vs v2 Comparison

**Date:** 2026-07-28

## Overall

| Metric | v1 | v2 | Delta |
|--------|----|----|-------|
| Accuracy | 0.6352 | 0.5841 | -0.0511 |
| F1 (macro) | 0.5375 | 0.5256 | -0.0119 |
| F1 (weighted) | 0.6415 | 0.6059 | -0.0356 |

## Per-Class Recall

| Class | v1 | v2 | Delta |
|-------|----|----|-------|
| Little or No Damage | 0.4366 | 0.5634 | +0.1268 |
| Mild Damage | 0.4444 | 0.4921 | +0.0477 |
| Severe Damage | 0.7500 | 0.6235 | -0.1265 |

## Gate Analysis

| | v1 | v2 |
|---|----|----|
| Image Gate | 0.1730 | 0.6441 |
| Text Gate | 0.0140 | 0.4314 |

## Recommendation

**NOT RECOMMENDED: v2 does not outperform v1.** Keep current checkpoint.

Delta F1: -0.0119 | Delta Little Recall: +0.1268 | Delta Mild Recall: +0.0477