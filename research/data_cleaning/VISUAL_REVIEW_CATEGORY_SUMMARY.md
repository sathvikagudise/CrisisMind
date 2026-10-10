# Visual Relevance Review — Category Summary

- Audit population: **3526** severity samples (full 3,526 subset)
- Removal candidates (conf≥0.50, CLIP irrelevant): **1037**
- Ambiguous (kept by default): **837**
- **Nothing has been removed yet.** Removal happens only after your confirmation.

| # | Folder / Category | Count | Sheets | Notes |
|---|-------------------|-------|--------|-------|
| 1 | `IRRELEVANT_SELFIE` | 13 | 1 |  |
| 2 | `IRRELEVANT_GROUP_PHOTO` | 2 | 1 |  |
| 3 | `IRRELEVANT_PORTRAIT` | 0 | 0 |  |
| 4 | `IRRELEVANT_UNRELATED_IMAGE` | 1 | 1 |  |
| 5 | `IRRELEVANT_NORMAL_SCENE` | 66 | 3 |  |
| 6 | `IRRELEVANT_NO_DISASTER_EVIDENCE` | 955 | 40 |  |
| 7 | `IMAGE_TEXT_MISMATCH` | 1389 | 58 | overlaps with folders 1-6 (948 candidates) and folder 8 (441 ambiguous); shown separately so you can judge text-image mismatch directly |
| 8 | `AMBIGUOUS_REVIEW` | 837 | 35 | kept by default; review only if you want to move any to removal |

### ID scheme
- `[cid]` = original row index in `VISUAL_REVIEW_CANDIDATES.csv` (removal candidates).
- `[A#n]` = index within the AMBIGUOUS_REVIEW list (kept by default).
- Labels on each cell: `id category conf image_text_relation | original_label | file`.

### Totals (with overlap)
- Images across all grouped sheets (images may repeat across folders): **3263**
- Unique removal-candidate images in folders 1-6: **1037**
- Unique ambiguous images in folder 8: **837**
- Unique images across folders 1-8: **sample_id              1874
image_path             1874
text                   1729
original_label            3
visual_category           7
visual_confidence      1445
text_disaster_prob     1418
image_text_relation       3
decision                  2
reason                 1380
dtype: int64** (non-KEEP)
