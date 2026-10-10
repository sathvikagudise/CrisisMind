# CrisisMind — Gate Collapse Analysis

**Date:** 2026-07-28

## Problem
v1 text gate ~0.014 (near zero). Text modality effectively ignored.

## Solution
Quadratic penalty: `L = weight * mean((gate - 0.5)^2)`

## Results

| Run | Reg Weight | Img Gate | Txt Gate | Diff |
|-----|-----------|----------|----------|------|
| focal_reg2 | 2.0 | 0.5351 | 0.4808 | 0.0543 |
| focal_reg1 | 1.0 | 0.5693 | 0.4410 | 0.1283 |
| wce_reg2 | 2.0 | 0.5653 | 0.4391 | 0.1262 |
| wce_reg1 | 1.0 | 0.6441 | 0.4314 | 0.2127 |
| focal_reg05 | 0.5 | 0.5990 | 0.4118 | 0.1872 |
| wce_reg05 | 0.5 | 0.7242 | 0.3647 | 0.3595 |
| focal_noreg | 0.0 | 0.9816 | 0.1379 | 0.8437 |

**Significant improvement.** Text gate 30.8x over v1.