import pandas as pd
import glob
import os
from backend.training.config import (
    ANNOTATIONS_DIR, DATA_IMAGE_DIR, DATASET_DIR, LABEL_MAP, LABEL_NAMES
)


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


def prepare_damage_data(dataframes):
    rows = []
    for event_name, df in dataframes.items():
        sub = df[["image_id", "image_damage", "image_path"]].copy()
        sub.dropna(subset=["image_damage"], inplace=True)
        sub = sub[sub["image_damage"] != "dont_know_or_cant_judge"]
        sub = sub[sub["image_damage"].isin(LABEL_MAP)]
        sub["label"] = sub["image_damage"].map(LABEL_MAP)
        sub["event"] = event_name
        sub["full_path"] = sub["image_path"].apply(
            lambda p: str((DATASET_DIR / p).resolve())
        )
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)
    combined.drop_duplicates(subset=["image_id"], inplace=True)
    return combined


def verify_images(df):
    existing_mask = df["full_path"].apply(os.path.exists)
    missing = df[~existing_mask]
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} images not found, dropping them.")
        df = df[existing_mask].copy()
    return df


def class_distribution(df):
    dist = df["label"].value_counts().sort_index()
    dist.index = dist.index.map(LABEL_NAMES)
    return dist
