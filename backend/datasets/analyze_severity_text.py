import pandas as pd
import glob
import os
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("dataset_analysis/CrisisMMD_v2.0")
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

LABEL_MAP = {
    "little_or_no_damage": 0,
    "mild_damage": 1,
    "severe_damage": 2,
}
LABEL_NAMES = {0: "little_or_no_damage", 1: "mild_damage", 2: "severe_damage"}


def load_tsvs():
    tsv_files = sorted(glob.glob(str(ANNOTATIONS_DIR / "*.tsv")))
    tsv_files = [f for f in tsv_files if not os.path.basename(f).startswith("._")]
    dataframes = {}
    for fp in tsv_files:
        name = os.path.basename(fp).replace("_final_data.tsv", "")
        df = pd.read_csv(fp, sep="\t", keep_default_na=True,
                         na_values=["", "nan", "NaN", None])
        dataframes[name] = df
    return dataframes


def main():
    print("=" * 60)
    print("  CrisisMMD Severity Dataset — Text Analysis")
    print("=" * 60)

    print("\n[1] Loading TSV files...")
    dataframes = load_tsvs()
    combined = pd.concat(dataframes.values(), ignore_index=True)
    print(f"  Total raw rows: {len(combined)}")

    print("\n[2] Filtering records with image_damage available...")
    sub = combined[["tweet_text", "image_damage"]].copy()
    sub.dropna(subset=["image_damage"], inplace=True)
    sub = sub[sub["image_damage"] != "dont_know_or_cant_judge"]
    sub = sub[sub["image_damage"].isin(LABEL_MAP)]
    sub["label"] = sub["image_damage"].map(LABEL_MAP)
    print(f"  Usable severity samples: {len(sub)}")

    print("\n[3] Text analysis...")

    empty_tweets = sub["tweet_text"].isna().sum() + (sub["tweet_text"] == "").sum()
    print(f"  Empty tweets:            {empty_tweets}")

    sub["tweet_text"] = sub["tweet_text"].fillna("")

    text_lengths = sub["tweet_text"].str.len()
    print(f"  Average tweet length:    {text_lengths.mean():.2f} chars")
    print(f"  Maximum tweet length:    {text_lengths.max()} chars")

    word_counts = sub["tweet_text"].str.split().str.len()
    word_counts = word_counts.fillna(0)
    print(f"  Average words per tweet: {word_counts.mean():.2f}")
    print(f"  Max words per tweet:     {int(word_counts.max())}")

    duplicate_mask = sub["tweet_text"].duplicated(keep=False)
    dup_count = duplicate_mask.sum()
    print(f"  Duplicate tweets:        {dup_count}")
    unique_mask = sub["tweet_text"].duplicated(keep="first")
    dup_to_remove = unique_mask.sum()
    print(f"  Duplicates to remove:    {dup_to_remove}")

    deduped = sub.drop_duplicates(subset=["tweet_text"], keep="first").copy()
    print(f"  Unique text samples:     {len(deduped)}")

    print("\n[4] Label distribution (after dedup) ...")
    dist = deduped["label"].value_counts().sort_index()
    total = len(deduped)
    print(f"  {'Label':30s} {'Count':>6s} {'%':>8s}")
    print(f"  {'-'*46}")
    for lbl, cnt in dist.items():
        print(f"  {LABEL_NAMES[lbl]:30s} {cnt:6d}  {100*cnt/total:7.2f}%")
    print(f"  {'-'*46}")
    print(f"  {'Total':30s} {total:6d}  {100.0:7.2f}%")

    print("\n[5] Label distribution (with all duplicates) ...")
    dist_raw = sub["label"].value_counts().sort_index()
    total_raw = len(sub)
    print(f"  {'Label':30s} {'Count':>6s} {'%':>8s}")
    print(f"  {'-'*46}")
    for lbl, cnt in dist_raw.items():
        print(f"  {LABEL_NAMES[lbl]:30s} {cnt:6d}  {100*cnt/total_raw:7.2f}%")
    print(f"  {'-'*46}")
    print(f"  {'Total':30s} {total_raw:6d}  {100.0:7.2f}%")

    print("\n[6] Sample tweets per class (deduped) ...")
    for lbl in [0, 1, 2]:
        samples = deduped[deduped["label"] == lbl]["tweet_text"].head(3)
        print(f"\n  --- {LABEL_NAMES[lbl]} (showing {len(samples)}) ---")
        for i, t in enumerate(samples, 1):
            print(f"  {i}. {t[:120]}..." if len(t) > 120 else f"  {i}. {t}")

    print("\n" + "=" * 60)
    print("  Text analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
