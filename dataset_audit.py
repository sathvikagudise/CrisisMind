import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("dataset_analysis/CrisisMMD_v2.0")
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
DATA_IMAGE_DIR = DATASET_DIR / "data_image"
REPORTS_DIR = Path("dataset_analysis/reports")

COLUMNS_OF_INTEREST = ["text_info", "image_info", "text_human", "image_human", "image_damage"]

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_tsvs():
    tsv_files = sorted(glob.glob(str(ANNOTATIONS_DIR / "*.tsv")))
    tsv_files = [f for f in tsv_files if not os.path.basename(f).startswith("._")]
    dataframes = {}
    for fp in tsv_files:
        name = os.path.basename(fp).replace("_final_data.tsv", "")
        df = pd.read_csv(fp, sep="\t", keep_default_na=True, na_values=["", "nan", "NaN", None])
        dataframes[name] = df
    return dataframes


def per_file_analysis(dataframes):
    rows = []
    for name, df in dataframes.items():
        row = {"File": name, "Rows": len(df), "Columns": len(df.columns)}
        for c in df.columns:
            row[f"missing_{c}"] = int(df[c].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def label_analysis(dataframes):
    records = []
    for name, df in dataframes.items():
        for col in COLUMNS_OF_INTEREST:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            counts = series.value_counts()
            total = len(series)
            for label, cnt in counts.items():
                records.append({
                    "Event": name,
                    "Column": col,
                    "Label": label,
                    "Count": cnt,
                    "Percentage": round(100.0 * cnt / total, 2),
                })
    return pd.DataFrame(records)


def verify_images(dataframes):
    missing = []
    existing = 0
    for name, df in dataframes.items():
        for path_val in df["image_path"].dropna().unique():
            full_path = DATASET_DIR / path_val
            if full_path.exists():
                existing += 1
            else:
                missing.append(str(path_val))
    return existing, missing


def count_aggregates(dataframes):
    all_images = set()
    all_tweets = set()
    severity_count = 0
    for df in dataframes.values():
        all_images.update(df["image_id"].dropna().unique())
        all_tweets.update(df["tweet_id"].dropna().unique())
        severity_count += df["image_damage"].dropna().shape[0]
    return len(all_images), len(all_tweets), severity_count


def plot_severity_distribution(dataframes):
    all_damage = []
    for df in dataframes.values():
        all_damage.extend(df["image_damage"].dropna().tolist())
    if not all_damage:
        return
    cnt = Counter(all_damage)
    labels, values = zip(*cnt.most_common())
    colors = sns.color_palette("Reds_r", len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=colors, edgecolor="black")
    ax.set_title("Damage Severity Distribution (All Events)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Severity Label")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "severity_distribution.png", dpi=150)
    plt.close(fig)


def plot_humanitarian_distribution(dataframes):
    all_human = []
    for df in dataframes.values():
        all_human.extend(df["text_human"].dropna().tolist())
        all_human.extend(df["image_human"].dropna().tolist())
    if not all_human:
        return
    cnt = Counter(all_human)
    labels, values = zip(*cnt.most_common())
    colors = sns.color_palette("viridis", len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    ax.set_title("Humanitarian Category Distribution (Text + Image)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Category")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(v), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "humanitarian_category_distribution.png", dpi=150)
    plt.close(fig)


def plot_eventwise_distribution(dataframes):
    events = []
    counts = []
    for name, df in dataframes.items():
        events.append(name.replace("_", " ").title())
        counts.append(len(df))
    colors = sns.color_palette("Set2", len(events))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(events, counts, color=colors, edgecolor="black")
    ax.set_title("Event-wise Sample Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eventwise_distribution.png", dpi=150)
    plt.close(fig)


def main():
    print("=" * 60)
    print("  CrisisMMD v2.0 — Dataset Audit")
    print("=" * 60)

    print("\n[1] Loading TSV files...")
    dataframes = load_tsvs()
    print(f"  Found {len(dataframes)} TSV files.")

    print("\n[2] Per-file analysis...")
    stats_df = per_file_analysis(dataframes)
    print(stats_df.to_string(index=False))

    print("\n[3] Label analysis on target columns...")
    label_df = label_analysis(dataframes)
    for col in COLUMNS_OF_INTEREST:
        subset = label_df[label_df["Column"] == col]
        if subset.empty:
            continue
        print(f"\n  --- {col} ---")
        overall = subset.groupby("Label").agg({"Count": "sum"}).reset_index()
        overall["Percentage"] = round(100.0 * overall["Count"] / overall["Count"].sum(), 2)
        overall = overall.sort_values("Count", ascending=False)
        for _, r in overall.iterrows():
            print(f"    {r['Label']:40s}  {r['Count']:>6d}  ({r['Percentage']:5.2f}%)")

    print("\n[4] Verifying image paths...")
    existing, missing = verify_images(dataframes)
    print(f"  Existing images found: {existing}")
    if missing:
        print(f"  Missing images: {len(missing)}")
        for m in missing[:10]:
            print(f"    - {m}")

    print("\n[5] Aggregate counts...")
    n_imgs, n_tweets, n_sev = count_aggregates(dataframes)
    print(f"  Total unique images:  {n_imgs}")
    print(f"  Total unique tweets:  {n_tweets}")
    print(f"  Total severity samples: {n_sev}")

    print("\n[6] Generating visualizations...")
    plot_severity_distribution(dataframes)
    plot_humanitarian_distribution(dataframes)
    plot_eventwise_distribution(dataframes)
    print(f"  Charts saved to: {REPORTS_DIR}/")

    print("\n[7] Generating summary report...")
    total_rows = sum(len(df) for df in dataframes.values())
    total_cols = len(next(iter(dataframes.values())).columns)
    total_labels = label_df.groupby("Column").agg({"Label": "nunique"}).to_dict()["Label"]

    with open("dataset_summary_report.md", "w", encoding="utf-8") as f:
        f.write("# CrisisMMD v2.0 — Dataset Summary Report\n\n")
        f.write(f"**Generated:** Auto-generated by `dataset_audit.py`\n\n")
        f.write("## 1. Dataset Overview\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Total TSV files | {len(dataframes)} |\n")
        f.write(f"| Total rows (samples) | {total_rows} |\n")
        f.write(f"| Total columns | {total_cols} |\n")
        f.write(f"| Unique tweets | {n_tweets} |\n")
        f.write(f"| Unique images | {n_imgs} |\n")
        f.write(f"| Severity-annotated samples | {n_sev} |\n\n")

        f.write("## 2. Per-File Statistics\n\n")
        f.write("| File | Rows | Columns |\n")
        f.write("|---|---|---|\n")
        for _, r in stats_df.iterrows():
            f.write(f"| {r['File']} | {r['Rows']} | {r['Columns']} |\n")
        f.write("\n**Column names:** ")
        f.write(", ".join(next(iter(dataframes.values())).columns.tolist()))
        f.write("\n\n")

        f.write("## 3. Missing Values (per file, per column)\n\n")
        f.write("| File | " + " | ".join(next(iter(dataframes.values())).columns) + " |\n")
        f.write("|" + "---|" * (len(next(iter(dataframes.values())).columns) + 1) + "\n")
        for _, r in stats_df.iterrows():
            vals = [str(r[f"missing_{c}"]) for c in next(iter(dataframes.values())).columns]
            f.write("| " + r["File"] + " | " + " | ".join(vals) + " |\n")
        f.write("\n")

        f.write("## 4. Label Distributions\n\n")
        for col in COLUMNS_OF_INTEREST:
            subset = label_df[label_df["Column"] == col]
            if subset.empty:
                continue
            f.write(f"### {col}\n\n")
            f.write("| Label | Total Count | Overall % |\n")
            f.write("|---|---|---|\n")
            overall = subset.groupby("Label").agg({"Count": "sum"}).reset_index()
            overall["Percentage"] = round(100.0 * overall["Count"] / overall["Count"].sum(), 2)
            overall = overall.sort_values("Count", ascending=False)
            for _, r in overall.iterrows():
                f.write(f"| {r['Label']} | {r['Count']} | {r['Percentage']:.2f}% |\n")
            f.write("\n")

        f.write("## 5. Image Path Verification\n\n")
        f.write(f"- **Existing image paths:** {existing}\n")
        f.write(f"- **Missing image paths:** {len(missing)}\n\n")

        f.write("## 6. Visualizations\n\n")
        f.write("### Severity Distribution\n")
        f.write("![Severity Distribution](dataset_analysis/reports/severity_distribution.png)\n\n")
        f.write("### Humanitarian Category Distribution\n")
        f.write("![Humanitarian Category](dataset_analysis/reports/humanitarian_category_distribution.png)\n\n")
        f.write("### Event-wise Distribution\n")
        f.write("![Event-wise Distribution](dataset_analysis/reports/eventwise_distribution.png)\n\n")

    print("  Report saved to: dataset_summary_report.md")
    print("\n" + "=" * 60)
    print("  Audit complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
