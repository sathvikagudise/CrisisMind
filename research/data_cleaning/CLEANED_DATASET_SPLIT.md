# Cleaned Dataset Split — Leak-Free Stratified 70/15/15

**Phase 4 deliverable.** Defines the train/validation/test split for the cleaned
population (3,105 samples) and guarantees that no perceptual near-duplicate image spans
more than one split.

## Method

1. **Population:** cleaned severity subset (3,105 samples) after Phase 3 selection.
2. **Group construction:** perceptual near-duplicate clusters (dHash Hamming distance
   ≤ 0.10, 64-bit) are collapsed into **indivisible groups** so all visually duplicate
   members are forced into the same split. Singletons are their own group.
   - 99 near-duplicate clusters among the cleaned population (220 samples), 2,885 singletons.
   - 2,984 groups total.
3. **Assignment:** deterministic greedy assignment minimizing the squared deviation of
   each split from its target class proportion (70/15/15), computed per severity class.
   Group order is seeded (random seed 42) then sorted by size (largest first) so the least
   flexible clusters are placed first. Ties broken by lowest cost.
4. **Reproducibility:** fixed seed 42, deterministic tie-breaking; the full algorithm is
   re-runnable and the assignment is persisted.

## Resulting split

| Split | Samples | little_or_no_damage | mild_damage | severe_damage | Share |
|---|---|---|---|---|---|
| Train | 2,173 | 282 | 482 | 1,409 | 70.0% |
| Validation | 467 | 61 | 104 | 302 | 15.0% |
| Test | 465 | 60 | 103 | 302 | 15.0% |
| **Total** | **3,105** | 403 | 689 | 2,013 | 100% |

### Per-class proportion achieved (target: 0.70 / 0.15 / 0.15)

| Class | Train | Val | Test |
|---|---|---|---|
| little_or_no_damage | 0.6998 | 0.1514 | 0.1489 |
| mild_damage | 0.6996 | 0.1509 | 0.1495 |
| severe_damage | 0.7000 | 0.1500 | 0.1500 |

Stratification deviation from target is ≤ 0.0004 in every cell.

## Leakage verification

| Check | Result |
|---|---|
| Near-duplicate clusters (≤0.10) spanning >1 split | **0** |
| Exact-identical image pairs spanning >1 split | **0** |
| All cluster members co-located in one split | verified per cluster |

This resolves the integrity issue found in the audit (§2.8), where 94 of 183 near-dup
clusters in the *original* split straddled the train/test boundary.

## Comparison vs original split

| Property | Original split | Cleaned split |
|---|---|---|
| Population | 3,526 | 3,105 |
| Train / Val / Test | 2,468 / 529 / 529 | 2,173 / 467 / 465 |
| Per-class stratification | stratified (no near-dup containment) | stratified + near-dup contained |
| Near-dup clusters crossing splits | 94 | **0** |
| Exact-identical pairs crossing splits | 27 | **0** |

## Artifacts

| Artifact | Location |
|---|---|
| Sample-level assignment | `research/data_cleaning/KEPT_SAMPLES.csv` (`split` column) |
| Train/val/test path lists | `dataset_analysis/CrisisMMD_v2.0_clean/splits/{train,val,test}_paths.txt` |
| Split metadata (JSON) | `dataset_analysis/CrisisMMD_v2.0_clean/splits/SPLIT_META.json` |
