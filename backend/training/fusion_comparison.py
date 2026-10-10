import numpy as np
import pandas as pd
from backend.training.config import LABEL_NAMES


def build_comparison_table(results):
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
            "Precision (macro)": f"{metrics.get('precision_macro', 0):.4f}",
            "Recall (macro)": f"{metrics.get('recall_macro', 0):.4f}",
            "F1 (macro)": f"{metrics.get('f1_macro', 0):.4f}",
            "F1 (weighted)": f"{metrics.get('f1_weighted', 0):.4f}",
        })
    df = pd.DataFrame(rows)
    best_idx = df["F1 (macro)"].astype(float).idxmax()
    return df, best_idx


def save_comparison_report(results, clf_reports, path):
    df, best_idx = build_comparison_table(results)
    best_model = df.iloc[best_idx]["Model"]

    with open(path, "w") as f:
        f.write("# Multimodal Fusion — Model Comparison Report\n\n")
        f.write("## Comparison Table\n\n")
        f.write(df.to_markdown(index=False) + "\n\n")
        f.write(f"**Best Model:** {best_model} (by Macro F1)\n\n")

        f.write("## Per-Class Performance\n\n")
        for name in results:
            if name not in clf_reports:
                continue
            f.write(f"### {name}\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|---|---|---|---|---|\n")
            for label_key in sorted(LABEL_NAMES):
                label_name = LABEL_NAMES[label_key]
                r = clf_reports[name].get(label_name, {})
                if r:
                    f.write(f"| {label_name} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |\n")
            f.write("\n")
