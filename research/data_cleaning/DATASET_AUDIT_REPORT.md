# CrisisMMD v2.0 — Dataset Audit Report

**Project:** CrisisMind (multimodal disaster-damage severity classification)
**Dataset:** CrisisMMD v2.0 (7 events, 2017 natural disasters)
**Scope of this audit:** The multimodal severity-classification pipeline subset used by the
Gated Fusion v3 model (3,526 image–text samples). The full CrisisMMD v2.0 corpus is larger
(18,082 annotation rows across all informative-class tasks); this audit focuses on the exact
sample population consumed by `backend/datasets/multimodal_data.py` → `load_multimodal_data()`,
which is the population the model is trained and evaluated on.

**Date:** 2026-08-14
**Status:** Complete

---

## 1. Population Under Audit

`load_multimodal_data()` selects rows whose **image** is:

- `image_info == "informative"` **and**
- `image_human == "infrastructure_and_utility_damage"` **and**
- `image_damage` is one of `{little_or_no_damage, mild_damage, severe_damage}`

and which have a non-empty cleaned text and a readable image file. This yields **3,526 samples**.

### 1.1 Label distribution

| Severity label | Count | Share |
|---|---|---|
| severe_damage | 2,212 | 62.7% |
| mild_damage | 839 | 23.8% |
| little_or_no_damage | 475 | 13.5% |
| **Total** | **3,526** | 100% |

The dataset is strongly class-imbalanced (≈4.7:1.8:1 severe:mild:little).

### 1.2 Event distribution

| Event | Count |
|---|---|
| hurricane_harvey | 892 |
| hurricane_maria | 862 |
| hurricane_irma | 795 |
| california_wildfires | 531 |
| mexico_earthquake | 178 |
| iraq_iran_earthquake | 173 |
| srilanka_floods | 95 |

### 1.3 Current train/val/test split (used by existing feature cache)

| Split | Count | little | mild | severe |
|---|---|---|---|---|
| Train | 2,468 | 333 | 587 | 1,548 |
| Val | 529 | 71 | 126 | 332 |
| Test | 529 | 71 | 126 | 332 |
| **Total** | **3,526** | 475 | 839 | 2,212 |

Split created with `sklearn.model_selection.train_test_split`, `random_state=42`, stratified
on label (70/15/15), via `backend/training/step1_extract.py`.

---

## 2. Data Quality Checks

### 2.1 File integrity (images)

| Check | Result |
|---|---|
| Annotation rows referencing a missing image file | **0** |
| Image files that fail to open/decode | **0** |
| Near-blank images (std < 4 at 64×64 after downscale) | **0** |
| Images with a minimum dimension < 64 px | **2** |
| Smallest image file on disk | 1,763 bytes |

The two sub-64 px images (both california_wildfires) are still decodable and represent
genuine (if low-resolution) content, so they are **not** removed.

### 2.2 Text integrity

| Check | Result |
|---|---|
| Rows with empty text after URL removal / whitespace collapse | **0** |
| Rows with heavy non-ASCII (≥30% of characters) | **0** |
| Rows containing any non-ASCII characters (emoji, accents) | 789 (normal for Twitter) |

### 2.3 Annotator confidence (`image_damage_conf`)

`image_damage_conf` is the CrisisMMD annotator-agreement / confidence score attached to each
severity label. Distribution across the 3,526 samples:

| Statistic | Value |
|---|---|
| Mean | 0.793 |
| Median | 0.832 |
| Min | 0.275 |
| Samples with conf == 1.0 | 1,569 |
| conf < 0.5 | 298 (8.5%) |
| conf < 0.6 | 335 (9.5%) |
| conf < 2/3 | 957 (27.1%) |
| conf < 0.7 | 1,698 (48.2%) |

Confidence by class (mean):

| Class | Mean conf |
|---|---|
| severe_damage | 0.848 |
| mild_damage | 0.692 |
| little_or_no_damage | 0.717 |

Low-confidence labels are concentrated in the minority classes, meaning the noisiest
annotations are precisely the samples the imbalanced model struggles most to learn.

### 2.4 Informative/humanitarian annotations

- All 3,526 rows have `image_info == "informative"` and
  `image_human == "infrastructure_and_utility_damage"` (this is a selection criterion,
  not a finding — confirmed 3,526 / 3,526).
- **398 rows** (11.3%) have `text_info == "not_informative"` **and**
  `text_human == "not_humanitarian"` — the two text conditions are set-identical.

  > **Decision (documented):** these rows are **not** removed. Severity classification is
  > image-grounded (`image_human`/`image_info` are already filtered), and a non-humanitarian
  > *text* does not invalidate the image label for this task. Removing them would be an
  > arbitrary, potentially metric-affecting choice without an objective justification tied
  > to label quality.

### 2.5 Text duplicates

Normalized text (lowercased, URLs stripped, whitespace collapsed):

| Statistic | Value |
|---|---|
| Rows sharing a duplicated normalized text | 530 (15.0%) |
| Distinct duplicated texts | 202 |
| Duplicate-text groups with conflicting severity labels | 104 |

Duplicate texts overwhelmingly arise from retweets. Text-duplicate rows whose *images* also
duplicate are handled by the image-duplicate passes (F/G); pure text duplicates with
distinct images carry independent visual evidence and are **not** removed on text grounds
alone.

### 2.6 Multiple images per tweet

| Statistic | Value |
|---|---|
| Tweets contributing >1 image to the subset | 202 |

### 2.7 Image near-duplicates (perceptual, dHash)

Perceptual hashing (64-bit dHash per image, Hamming distance):

| Statistic | Value |
|---|---|
| Exact-identical image groups (Hamming distance == 0) | **68** |
| Near-duplicate clusters (Hamming distance ≤ 0.10) | **183** |

### 2.8 Train/test leakage via near-duplicates ⚠️

| Leakage check | Clusters spanning >1 split |
|---|---|
| Near-duplicate clusters (≤0.10) spanning train/val/test | **94 of 183** |
| Exact-identical groups spanning train/val/test | **27 of 68** |

This is the most significant integrity finding: near-identical images (sometimes the exact
same photograph) appear on **both sides of the train/test boundary**. Any evaluation metric
computed on the existing split is optimistically biased, because the model can memorize
near-duplicates seen during training and be credited for "generalizing" to their test-time
copies.

---

## 3. Root-Cause Summary

1. **Perceptual duplicates + split leakage.** Real-world disaster imagery is heavily
   re-shared (news agencies, retweets, re-posts); 183 near-duplicate image clusters exist,
   94 of them straddle the current train/test boundary. The existing 0.70/0.15/0.15 split
   does not protect against this.
2. **Noisy minority-class labels.** 9.5% of samples have `image_damage_conf < 0.6`, and
   these concentrate in mild/little classes where the model is weakest.
3. **Conflicting annotations on near-duplicates.** Several near-dup clusters carry
   *different* severity labels for visually identical imagery, which is annotation noise,
   not a real signal the model can learn.
4. **Imbalanced classes.** 62.7% of samples are severe_damage; the model already handles
   these well and the minority classes carry most of the label noise.

---

## 4. Objective Cleaning Criteria (Phase 2)

Every rule below is **deterministic, threshold-based, and independent of model predictions**.
No sample is removed because a model is "confused" by it; no rule is tuned to move any metric.
All rules act on annotation metadata or on perceptual hashing of the images, never on model
output.

| Rule | Category | Criterion | Count removed |
|---|---|---|---|
| **A** | Unreadable/missing image | File missing or fails to decode | 0 |
| **B** | Blank image | Downscaled 64×64 std < 4 | 0 |
| **C** | Tiny image | Any dimension < 64 px | 0 |
| **D** | Empty text | Empty after URL strip + collapse | 0 |
| **E** | Non-informative text | `text_info == "not_informative"` | 0* |
| **F** | Exact duplicate image | dHash Hamming distance == 0.0 within a group → keep the highest-confidence member, drop the rest | 83 |
| **G** | Conflicting near-duplicate labels | dHash Hamming distance ≤ 0.10 cluster with >1 distinct severity label → majority-label wins; ties keep the single highest-confidence member | 25 |
| **H** | Ambiguous severity label | `image_damage_conf < 0.6` (weak annotator agreement) | 313 |
| | **Total removed** | | **421 (11.9%)** |

\* Documented decision: kept intentionally (see §2.4); the rule exists so the choice is
explicit and auditable.

Rationale for threshold choices:

- **0.6 on `image_damage_conf`**: a hard decision boundary at 0.6 removes the clearly
  ambiguous tail (9.5% of samples, mean 0.46, worst 0.28) while retaining 90.5% of the
  population with a mean conf of 0.837. It is a single, pre-registered threshold — not
  searched over to maximize any score.
- **dHash ≤ 0.10** as near-duplicate: standard perceptual-duplicate tolerance; with 64 bits
  this corresponds to ≥54/64 bits agreeing.
- **Tie-breaks by confidence**: in F and G-tie cases, "highest `image_damage_conf`" is the
  only objective selector available from the annotation metadata.

### 4.1 Resulting cleaned population

| | Original | After cleaning | Δ |
|---|---|---|---|
| Total samples | 3,526 | **3,105** | −421 (−11.9%) |
| severe_damage | 2,212 | 2,013 | −199 |
| mild_damage | 839 | 689 | −150 |
| little_or_no_damage | 475 | 403 | −72 |
| Mean conf (kept) | 0.793 | **0.837** | +0.044 |
| Min conf (kept) | 0.275 | **0.602** | +0.327 |

Cleaned class distribution: severe 64.8% / mild 22.2% / little 13.0% — still imbalanced,
but with the noisy tail removed from every class.

---

## 5. Methodology / Reproducibility

All analysis scripts are deterministic:

- Perceptual hashes: 64-bit dHash (grayscale → 9×8 → diff → 64 bits) per image, computed
  with a fixed pipeline; hash matrix saved as `hashes.npy` with `hash_paths.json`.
- Duplicate clustering: pairwise Hamming distance, undirected graph, connected components.
- Confidence thresholds and split seed (42) are fixed constants.
- The removal plan is recorded row-by-row in `REMOVED_SAMPLES.csv` (see Phase 3) so every
  decision can be audited.

## 6. Phase 3+ Preview

- **Phase 3** — persist the audited original and cleaned image directories, and emit
  `REMOVED_SAMPLES.csv` (full traceability) + `CLEANING_SUMMARY.md`.
- **Phase 4** — recreate the 70/15/15 stratified split on the *cleaned* population with
  near-duplicate containment so that no near-duplicate cluster straddles train/test, then
  document it in `CLEANED_DATASET_SPLIT.md`.
- **Phase 5–7** — retrain Gated Fusion v3 (same architecture, same hyperparameter recipe
  as the published `wide_wce_g1.0_lr1e-04` run) on the cleaned data and compare.
