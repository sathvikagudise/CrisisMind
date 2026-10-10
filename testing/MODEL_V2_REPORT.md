# CrisisMind — Model v2 Training Report

**Date:** 2026-07-28

## Training Configuration

- Architecture: EfficientNet-B0 + DistilBERT + GatedFusion
- Hybrid-cache: frozen encoders, trainable fusion head
- Balanced sampling: WeightedRandomSampler
- Gate regularization: quadratic penalty toward 0.5
- Optimizer: AdamW + cosine annealing
- Max 40 epochs, early stopping (patience=7)
- Selection: Validation Macro F1

## Loss Comparison

| Run | Loss | Gate Reg | F1(m) | F1(w) | Acc | R-L | R-M | R-S |
|-----|------|----------|-------|-------|-----|-----|-----|-----|
| wce_reg1 ** | wce | 1.0 | 0.5256 | 0.6059 | 0.5841 | 0.5634 | 0.4921 | 0.6235 |
| focal_noreg | focal | 0.0 | 0.4864 | 0.5660 | 0.5350 | 0.4930 | 0.5476 | 0.5392 |
| wce_reg05 | wce | 0.5 | 0.4800 | 0.5389 | 0.5085 | 0.6761 | 0.5079 | 0.4729 |
| focal_reg2 | focal | 2.0 | 0.4579 | 0.5105 | 0.4858 | 0.4225 | 0.7222 | 0.4096 |
| wce_reg2 | wce | 2.0 | 0.4515 | 0.5315 | 0.4934 | 0.5493 | 0.4762 | 0.4880 |
| focal_reg1 | focal | 1.0 | 0.4482 | 0.5214 | 0.4839 | 0.6197 | 0.4603 | 0.4639 |
| focal_reg05 | focal | 0.5 | 0.4218 | 0.4660 | 0.4386 | 0.6197 | 0.5317 | 0.3645 |

## Best: wce_reg1

- F1 (macro): 0.5256
- Accuracy: 0.5841
- Gates: img=0.6441 txt=0.4314

### Per-Class

| Class | Prec | Recall | F1 |
|-------|------|--------|----|
| Little or No Damage | 0.3738 | 0.5634 | 0.4494 |
| Mild Damage | 0.3605 | 0.4921 | 0.4161 |
| Severe Damage | 0.8280 | 0.6235 | 0.7113 |