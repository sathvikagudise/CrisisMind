"""Phase 0: Full CrisisMMD v2.0 investigation.

Profiles all 18,082 annotation rows:
  1. total rows
  2. rows with valid image_damage labels
  3. rows without image_damage labels
  4. other useful annotations (text_info, text_human, image_human, image_info)
  5. which unlabeled samples could contribute to SSL / pretraining
  8. samples irrelevant to visual damage assessment

Outputs: research/final_experiment/phase0_full_dataset_investigation.json
         research/final_experiment/PHASE0_FULL_DATASET_INVESTIGATION.md
"""
import json
import glob
import os
from collections import Counter
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN = os.path.join(ROOT, "dataset_analysis", "CrisisMMD_v2.0", "annotations")
OUT = os.path.join(ROOT, "research", "final_experiment")

VALID_DAMAGE = {"little_or_no_damage", "mild_damage", "severe_damage"}

files = sorted(glob.glob(os.path.join(ANN, "*.tsv")))
files = [f for f in files if not os.path.basename(f).startswith("._")]
dfs = []
for fp in files:
    df = pd.read_csv(fp, sep="\t", keep_default_na=True)
    df["event"] = os.path.basename(fp).replace("_final_data.tsv", "")
    dfs.append(df)
full = pd.concat(dfs, ignore_index=True)

res = {}
res["total_annotation_rows"] = int(len(full))
res["unique_images"] = int(full["image_path"].nunique())
res["unique_tweets"] = int(full["tweet_id"].nunique())

# --- damage label availability -------------------------------------------------
dmg = full["image_damage"].copy()
res["rows_with_valid_image_damage"] = int(dmg.isin(VALID_DAMAGE).sum())
res["rows_without_image_damage_nan"] = int(dmg.isna().sum())
res["rows_with_dont_know_cant_judge"] = int((dmg == "dont_know_or_cant_judge").sum())
res["image_damage_value_counts_raw"] = {str(k): int(v) for k, v in dmg.value_counts(dropna=False).items()}

# cross-tab: what other labels do non-damage rows carry?
nodmg = full[~dmg.isin(VALID_DAMAGE)]
res["unlabelled_rows_profile"] = {
    "total": int(len(nodmg)),
    "with_text": int((nodmg["tweet_text"].fillna("").str.len() > 0).sum()),
    "image_info_distribution": {str(k): int(v) for k, v in nodmg["image_info"].value_counts(dropna=False).items()},
    "image_human_distribution": {str(k): int(v) for k, v in nodmg["image_human"].value_counts(dropna=False).items()},
    "text_info_distribution": {str(k): int(v) for k, v in nodmg["text_info"].value_counts(dropna=False).items()},
    "text_human_distribution": {str(k): int(v) for k, v in nodmg["text_human"].value_counts(dropna=False).items()},
}

# labelled subset profile
lab = full[dmg.isin(VALID_DAMAGE)].copy()
res["labelled_rows_profile"] = {
    "total": int(len(lab)),
    "severity_distribution": {str(k): int(v) for k, v in lab["image_damage"].value_counts().items()},
    "image_info_distribution": {str(k): int(v) for k, v in lab["image_info"].value_counts(dropna=False).items()},
    "image_human_distribution": {str(k): int(v) for k, v in lab["image_human"].value_counts(dropna=False).items()},
    "per_event": {e: int(g["image_damage"].isin(VALID_DAMAGE).sum()) for e, g in lab.groupby("event")},
}

# how many labelled-by-damage rows exist OUTSIDE the current pipeline filter?
cur = lab[(lab["image_info"] == "informative") &
          (lab["image_human"] == "infrastructure_and_utility_damage")]
res["damage_labelled_rows_current_filter"] = int(len(cur))
res["damage_labelled_rows_outside_current_filter"] = {
    str(k): int(v) for k, v in lab.loc[~lab.index.isin(cur.index), "image_human"].value_counts(dropna=False).items()
}

# severity distribution for image-human categories outside infrastructure damage
outside = lab[~lab.index.isin(cur.index)]
res["outside_current_filter_severity"] = {
    str(h): {str(s): int(c) for s, c in g["image_damage"].value_counts().items()}
    for h, g in outside.groupby("image_human", dropna=False)
}

# duplicate image paths across the whole corpus (same image annotated multiple times?)
dup_img = full.groupby("image_path").size()
res["images_annotated_more_than_once"] = int((dup_img > 1).sum())
res["max_annotations_per_image"] = int(dup_img.max())

# conflicting damage labels on same image path
conf = 0
for p, g in full[dmg.isin(VALID_DAMAGE)].groupby("image_path"):
    if g["image_damage"].nunique() > 1:
        conf += 1
res["images_with_conflicting_damage_labels"] = int(conf)

json.dump(res, open(os.path.join(OUT, "phase0_full_dataset_investigation.json"), "w"), indent=2)
print(json.dumps(res, indent=2)[:6000])
