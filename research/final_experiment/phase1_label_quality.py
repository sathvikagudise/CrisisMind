"""Phase 1: Label-quality investigation + human-review lists + train-expansion safety.

Outputs (research/final_experiment/):
  HUMAN_REVIEW_PRIORITY.csv     - severity-labelled KEPT samples ranked for human review
  EXPANSION_CANDIDATES.csv      - infra-damage rows lacking severity + don't-know rows (future human labelling)
  RECLAIM_TRAIN_ONLY.csv        - rule-H (conf 0.275-0.6) removed samples that are SAFE to add to TRAIN ONLY
                                  (no perceptual near-duplicate vs val/test at Hamming<=0.10)
  PHASE1_LABEL_QUALITY.json     - machine-readable summary
No label is modified here. No sample is removed here. Review lists only.
"""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "final_experiment")
DATASET = os.path.join(ROOT, "dataset_analysis", "CrisisMMD_v2.0")
TMP_HASHES = r"C:/Users/sathv/AppData/Local/Temp/opencode"

kept = pd.read_csv(os.path.join(ROOT, "research", "data_cleaning", "KEPT_SAMPLES.csv"))
removed = pd.read_csv(os.path.join(ROOT, "research", "data_cleaning", "REMOVED_SAMPLES.csv"))
audit = pd.read_csv(os.path.join(ROOT, "research", "data_cleaning", "VISUAL_RELEVANCE_AUDIT.csv"))
opt_split = pd.read_csv(os.path.join(ROOT, "research", "optimization", "cleaned_dataset.csv"))

# The authoritative split is the optimization run's split (train 2209 / val 460 / test 436)
split_map = opt_split.set_index("image_path")["split"].to_dict()
kept["split"] = kept["image_path"].map(split_map)
assert kept["split"].notna().all()

audit_cols = ["decision", "visual_category", "visual_confidence", "text_disaster_prob",
              "image_text_relation"]
for c in audit_cols:
    kept[c] = kept["image_path"].map(audit.set_index("image_path")[c])

res = {"n_kept": int(len(kept)),
       "split_sizes": kept["split"].value_counts().to_dict()}

# ------------------------------------------------------------------
# 1. Suspicion scoring for human review of the KEPT severity data
# ------------------------------------------------------------------
def suspicion(r):
    score = 0
    tags = []
    if r["decision"] == "REVIEW_REMOVE":
        score += 2; tags.append("clip_review_remove")
    elif r["decision"] == "AMBIGUOUS_REVIEW":
        score += 1; tags.append("clip_ambiguous")
    if r["visual_category"].startswith("IRRELEVANT"):
        score += 1; tags.append(f"clip_{r['visual_category']}")
    if r["image_text_relation"] == "IMAGE_TEXT_MISMATCH":
        score += 1; tags.append("clip_image_text_mismatch")
    c = float(r["conf"])
    if c < 0.667:
        score += 2; tags.append("low_annotator_conf(<2/3)")
    elif c < 0.75:
        score += 1; tags.append("borderline_conf(<0.75)")
    return score, ";".join(tags)

sus = kept.apply(lambda r: pd.Series(suspicion(r), index=["suspicion_score", "suspicion_tags"]), axis=1)
kept = pd.concat([kept, sus], axis=1)

review = kept[kept["suspicion_score"] > 0].sort_values(
    ["suspicion_score", "conf"], ascending=[False, True])
review_out = review[[
    "image_path", "tweet_text", "image_damage", "conf", "split",
    "suspicion_score", "suspicion_tags", "visual_category", "image_text_relation"]]
review_out.to_csv(os.path.join(OUT, "HUMAN_REVIEW_PRIORITY.csv"), index=False)
res["human_review_counts"] = {
    "total_flagged": int(len(review_out)),
    "tier1_score>=4": int((review_out.suspicion_score >= 4).sum()),
    "tier2_score_2_3": int(review_out.suspicion_score.between(2, 3).sum()),
    "tier3_score_1": int((review_out.suspicion_score == 1).sum()),
    "flagged_in_test": int((review_out.split == "test").sum()),
}

# Text-keyword contradiction heuristics -> additional review flags (not corrections)
kw_neg = ["no damage", "not damaged", "minor damage", "little damage", "survived", "safe"]
severe_kw_hits = []
for _, r in kept[(kept.image_damage == "severe_damage")].iterrows():
    t = str(r.tweet_text).lower()
    if any(k in t for k in kw_neg):
        severe_kw_hits.append(r.image_path)
kept.loc[kept.image_path.isin(severe_kw_hits), "suspicion_tags"] += ";text_contradiction_heuristic"
res["severe_text_contradiction_candidates"] = len(severe_kw_hits)

# ------------------------------------------------------------------
# 2. Expansion candidates (rows that could gain severity labels via HUMAN verification)
# ------------------------------------------------------------------
import glob
files = sorted(glob.glob(os.path.join(DATASET, "annotations", "*.tsv")))
files = [f for f in files if not os.path.basename(f).startswith("._")]
rows = []
for fp in files:
    df = pd.read_csv(fp, sep="\t", keep_default_na=True)
    df["event"] = os.path.basename(fp).replace("_final_data.tsv", "")
    rows.append(df)
full = pd.concat(rows, ignore_index=True)
infra_nosev = full[(full.image_human == "infrastructure_and_utility_damage") & (full.image_damage.isna())]
dk = full[full.image_damage == "dont_know_or_cant_judge"]
exp = pd.concat([infra_nosev.assign(candidate_type="infrastructure_damage_no_severity_label"),
                 dk.assign(candidate_type="annotators_could_not_judge")])
exp[["event", "image_path", "tweet_text", "image_human", "image_damage", "candidate_type"]].to_csv(
    os.path.join(OUT, "EXPANSION_CANDIDATES.csv"), index=False)
res["expansion_candidates"] = {
    "infrastructure_no_severity": int(len(infra_nosev)),
    "cant_judge": int(len(dk)),
    "total": int(len(exp)),
}

# ------------------------------------------------------------------
# 3. Reclaim safety analysis: rule-H removed samples -> TRAIN ONLY if safe
# ------------------------------------------------------------------
h_removed = removed[removed.removal_category == "H"].copy()
print(f"Rule-H removed candidates: {len(h_removed)}")

hp = os.path.join(TMP_HASHES, "hash_paths.json")
if os.path.exists(hp):
    hash_paths = json.load(open(hp))
    hashes = np.load(os.path.join(TMP_HASHES, "hashes.npy"))
    h2i = {p: i for i, p in enumerate(hash_paths)}
else:
    h2i = {}

def dhash(img, size=16):
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(g, dtype=np.int16)
    d = px[:, 1:] > px[:, :-1]
    bits = np.packbits(d.flatten())
    return bits

def hamming_bits(a, b):
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())

# hashes stored as 256-bit? verify shape
hash_bit_len = None
if h2i:
    hash_bit_len = hashes.shape[1] * 8

valtest_paths = opt_split[opt_split.split.isin(["val", "test"])]["image_path"].tolist()
valtest_hashes = {}
missing_ht = 0
for p in valtest_paths:
    if p in h2i:
        valtest_hashes[p] = hashes[h2i[p]]
    else:
        missing_ht += 1

# Repo convention (run_all_phases.py): distance = mean(byte inequality), near-dup <= 0.10
safe_rows, unsafe_rows = [], []
vt_matrix = np.stack(list(valtest_hashes.values())) if valtest_hashes else None
for _, r in h_removed.iterrows():
    p = r["image_path"]
    if vt_matrix is None or p not in h2i:
        unsafe_rows.append((p, "no_hash")); continue
    hv = hashes[h2i[p]]
    d = (vt_matrix != hv[None, :]).mean(axis=1)
    min_d = float(d.min()) if len(d) else 1.0
    if min_d <= 0.10:
        unsafe_rows.append((p, f"near_dup_valtest(min={min_d:.3f})"))
    else:
        safe_rows.append((p, f"safe(min={min_d:.3f})"))

safe_set = {p for p, _ in safe_rows}
reclaim = h_removed[h_removed.image_path.isin(safe_set)].copy()
reclaim["assigned_split"] = "TRAIN_ONLY_expanded"
reclaim.to_csv(os.path.join(OUT, "RECLAIM_TRAIN_ONLY.csv"), index=False)

pd.DataFrame(unsafe_rows, columns=["image_path", "reason"]).to_csv(
    os.path.join(OUT, "RECLAIM_UNSAFE.csv"), index=False)

res["reclaim_analysis"] = {
    "rule_h_total": int(len(h_removed)),
    "safe_for_train_only": int(len(reclaim)),
    "unsafe_near_dup_or_missing": len(unsafe_rows),
    "reclaim_class_distribution": reclaim["original_label"].value_counts().to_dict(),
}

json.dump(res, open(os.path.join(OUT, "PHASE1_LABEL_QUALITY.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
