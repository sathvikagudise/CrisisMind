# CrisisMind — Manual Evaluation

**Date:** 2026-07-28

## Text-Only Evaluation

| # | Text (truncated) | True | v1 Pred | v2 Pred | v3_w Pred | v1 OK? | v2 OK? | v3 OK? |
|---|-----------------|------|---------|---------|-----------|--------|--------|--------|
| 1 | The road has a few visible cracks but tr... | Lit | Lit | Lit | Lit | Y | Y | Y |
| 2 | Some tree branches fell on the lawn but ... | Lit | Lit | Lit | Lit | Y | Y | Y |
| 3 | Light rain caused minor puddles on the s... | Lit | Mil | Mil | Mil | N | N | N |
| 4 | A few roof tiles shifted slightly during... | Lit | Mil | Mil | Mil | N | N | N |
| 5 | The fence has some minor scratches from ... | Lit | Mil | Lit | Mil | N | Y | N |
| 6 | Several buildings have broken windows an... | Mil | Lit | Lit | Mil | N | N | Y |
| 7 | Floodwater has entered the ground floor ... | Mil | Sev | Mil | Mil | N | Y | Y |
| 8 | The church roof collapsed partially due ... | Mil | Mil | Mil | Mil | Y | Y | Y |
| 9 | Moderate cracks in the retaining wall al... | Mil | Lit | Lit | Lit | N | N | N |
| 10 | Three vehicles damaged by falling tree b... | Mil | Mil | Mil | Mil | Y | Y | Y |
| 11 | Multiple buildings have collapsed and re... | Sev | Sev | Sev | Sev | Y | Y | Y |
| 12 | Entire neighbourhood destroyed by the wi... | Sev | Sev | Sev | Sev | Y | Y | Y |
| 13 | The hospital has severe structural damag... | Sev | Lit | Lit | Sev | N | N | Y |
| 14 | Hundreds of homes washed away by the cat... | Sev | Sev | Mil | Mil | Y | N | N |
| 15 | Major earthquake has reduced the city ce... | Sev | Sev | Sev | Sev | Y | Y | Y |

**Accuracy:** v1=8/15 (53%)  v2=9/15 (60%)  v3=10/15 (67%)

## Image-Only Evaluation (15 images)

| Image | v1 Pred | v1 ImgG | v2 Pred | v2 ImgG | v3 Pred | v3 ImgG |
|-------|---------|---------|---------|---------|---------|---------|
| 870283105151668224_1.jpg | Sev | 0.25 | Mil | 0.49 | Mil | 0.39 |
| 901805287958601728_0.jpg | Sev | 0.13 | Sev | 0.48 | Sev | 0.45 |
| 901835409440768000_1.jpg | Lit | 0.34 | Lit | 0.52 | Lit | 0.50 |
| 905230326359891968_0.jpg | Lit | 0.15 | Lit | 0.46 | Lit | 0.48 |
| 905686913876582401_0.jpg | Sev | 0.16 | Lit | 0.44 | Mil | 0.41 |
| 906664753602457601_0.jpg | Sev | 0.43 | Sev | 0.48 | Mil | 0.45 |
| 907638041661108224_0.jpg | Lit | 0.27 | Mil | 0.49 | Mil | 0.50 |
| 909740728909205504_0.jpg | Sev | 0.43 | Sev | 0.51 | Mil | 0.48 |
| 909742125662789632_0.jpg | Lit | 0.17 | Mil | 0.41 | Mil | 0.44 |
| 910110509852184576_0.jpg | Lit | 0.29 | Lit | 0.46 | Lit | 0.46 |
| 910162829461860352_0.jpg | Sev | 0.48 | Sev | 0.53 | Sev | 0.44 |
| 910559723770040326_1.jpg | Sev | 0.24 | Sev | 0.47 | Mil | 0.43 |
| 910691589034409984_0.jpg | Mil | 0.24 | Mil | 0.49 | Mil | 0.49 |
| 920016499473047552_0.jpg | Lit | 0.26 | Lit | 0.47 | Mil | 0.42 |
| 920131252602142720_0.jpg | Sev | 0.24 | Sev | 0.50 | Sev | 0.45 |

## Multimodal Evaluation (image + text combined)

| Image | True | v1 Pred | v2 Pred | v3 Pred | v1 Gate | v2 Gate | v3 Gate |
|-------|------|---------|---------|---------|---------|---------|---------|
| 870283105151668224_1.jpg | Mil | Sev | Mil | Mil | I=0.14T=0.02 | I=0.60T=0.54 | I=0.47T=0.47 |
| 901805287958601728_0.jpg | Lit | Sev | Sev | Sev | I=0.07T=0.01 | I=0.58T=0.48 | I=0.49T=0.45 |
| 901835409440768000_1.jpg | Mil | Lit | Lit | Lit | I=0.29T=0.01 | I=0.65T=0.53 | I=0.57T=0.48 |
| 905230326359891968_0.jpg | Lit | Lit | Lit | Lit | I=0.07T=0.04 | I=0.61T=0.60 | I=0.51T=0.50 |
| 905686913876582401_0.jpg | Lit | Mil | Lit | Lit | I=0.09T=0.01 | I=0.57T=0.62 | I=0.46T=0.50 |
| 906664753602457601_0.jpg | Lit | Mil | Lit | Mil | I=0.21T=0.04 | I=0.56T=0.62 | I=0.48T=0.50 |
| 907638041661108224_0.jpg | Lit | Lit | Lit | Lit | I=0.14T=0.01 | I=0.63T=0.51 | I=0.55T=0.47 |
| 909740728909205504_0.jpg | Lit | Sev | Mil | Mil | I=0.24T=0.04 | I=0.63T=0.45 | I=0.54T=0.45 |
| 909742125662789632_0.jpg | Lit | Lit | Lit | Lit | I=0.07T=0.02 | I=0.54T=0.69 | I=0.49T=0.55 |
| 910110509852184576_0.jpg | Lit | Lit | Lit | Lit | I=0.15T=0.02 | I=0.60T=0.66 | I=0.50T=0.50 |
| 910162829461860352_0.jpg | Lit | Sev | Sev | Sev | I=0.30T=0.03 | I=0.66T=0.40 | I=0.51T=0.47 |
| 910559723770040326_1.jpg | Mil | Sev | Sev | Mil | I=0.18T=0.02 | I=0.61T=0.52 | I=0.50T=0.49 |
| 910691589034409984_0.jpg | Lit | Mil | Mil | Mil | I=0.10T=0.03 | I=0.65T=0.55 | I=0.53T=0.51 |
| 920016499473047552_0.jpg | Lit | Lit | Lit | Lit | I=0.12T=0.02 | I=0.55T=0.64 | I=0.49T=0.53 |
| 920131252602142720_0.jpg | Lit | Lit | Lit | Lit | I=0.10T=0.02 | I=0.57T=0.56 | I=0.50T=0.50 |

## Summary

| Metric | v1 | v2 | v3_wide |
|--------|----|----|---------|
| Text-only accuracy | 8/15 (53%) | 9/15 (60%) | 10/15 (67%) |
| Gate collapse | Yes (T=0.014) | Fixed (T=0.431) | Fixed (T=0.436) |
| Architecture | Standard | Standard | Wide (1024, BN) |