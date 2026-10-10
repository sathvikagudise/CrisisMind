# CrisisMind — v1 vs v2 vs v3 Comparison

**Date:** 2026-07-28

## Overall

| Metric | v1 | v2 | v3 | v3-v1 | v3-v2 |
|--------|----|----|-----|-------|-------|
| Accuracy | 0.6352 | 0.5766 | 0.6011 | -0.0341 | +0.0245 |
| F1 (macro) | 0.5375 | 0.5256 | 0.5372 | -0.0003 | +0.0116 |
| F1 (weighted) | 0.6415 | 0.6059 | 0.6220 | -0.0195 | +0.0161 |

## Per-Class Recall

| Class | v1 | v2 | v3 | v3-v1 | v3-v2 |
|-------|----|----|-----|-------|-------|
| Little or No Damage | 0.4366 | 0.5634 | 0.4648 | +0.0282 | -0.0986 |
| Mild Damage | 0.4444 | 0.4921 | 0.5635 | +0.1191 | +0.0714 |
| Severe Damage | 0.7500 | 0.6235 | 0.6446 | -0.1054 | +0.0211 |

## Gate Analysis

| | v1 | v2 | v3 |
|---|----|----|----|
| Image Gate | 0.1730 | 0.6440 | 0.5365 |
| Text Gate | 0.0140 | 0.4310 | 0.4357 |

## Version History

| Version | Key Change | F1(macro) | Text Gate |
|---------|-----------|-----------|-----------|
| v1 | Baseline production | 0.5375 | 0.014 (collapsed) |
| v2 | Gate regularization + balanced sampling | 0.5256 | 0.431 |
| v3 | Wider fusion head + optimized hyperparams | 0.5372 | 0.436 |