import pandas as pd
import glob
import os
import re
from pathlib import Path
from backend.training.config import DATASET_DIR, ANNOTATIONS_DIR, DATA_IMAGE_DIR, LABEL_MAP


def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_multimodal_data():
    tsv_files = sorted(glob.glob(str(ANNOTATIONS_DIR / "*.tsv")))
    tsv_files = [f for f in tsv_files if not os.path.basename(f).startswith("._")]
    rows = []
    for fp in tsv_files:
        df = pd.read_csv(fp, sep="\t", keep_default_na=True,
                         na_values=["", "nan", "NaN", None])
        sub = df[["tweet_text", "image_damage", "image_path"]].copy()
        sub.dropna(subset=["image_damage", "image_path"], inplace=True)
        sub = sub[sub["image_damage"] != "dont_know_or_cant_judge"]
        sub = sub[sub["image_damage"].isin(LABEL_MAP)]
        sub["label"] = sub["image_damage"].map(LABEL_MAP)
        sub["tweet_text"] = sub["tweet_text"].fillna("")
        sub["tweet_text"] = sub["tweet_text"].apply(clean_tweet)
        sub["full_image_path"] = sub["image_path"].apply(
            lambda p: str((DATASET_DIR / p).resolve())
        )
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)
    combined = combined[combined["tweet_text"].str.len() > 0].copy()
    combined = combined[combined["full_image_path"].apply(os.path.exists)].copy()
    combined.drop_duplicates(subset=["tweet_text", "image_path"], keep="first", inplace=True)
    return combined
