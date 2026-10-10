"""Phase 6: Full evaluation + latency measurement for the cleaned v3 model.

Reads cleaned_v3_wide_wce_g1.0_lr1e-04_best.pth, re-runs eval on the clean test split,
measures fusion-head inference latency (CPU, steady-state mean of 3 trials), and emits:
  - backend/outputs/checkpoints/v3_clean/cleaned_v3_wide_wce_g1.0_lr1e-04_eval.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

from backend.training.config import LABEL_NAMES, NUM_CLASSES, CHECKPOINTS_DIR, RANDOM_SEED
from backend.models.fusion_models import GatedFusionWide, count_parameters
from backend.config.shared import DEVICE

SEVERITY_LABELS = ["Little or No Damage", "Mild Damage", "Severe Damage"]
DEV = torch.device(DEVICE)
OUT_DIR = CHECKPOINTS_DIR / "v3_clean"
RUN_NAME = "cleaned_v3_wide_wce_g1.0_lr1e-04"
CACHE_PATH = OUT_DIR / "feature_cache.pt"
EVAL_BATCH = 64


def eval_full(model, loader):
    model.eval()
    preds_l, labels_l = [], []
    t0 = time.time()
    with torch.no_grad():
        for img_f, txt_f, labels in loader:
            img_f, txt_f, labels = img_f.to(DEV), txt_f.to(DEV), labels.to(DEV)
            logits, _ = model(img_f, txt_f)
            preds_l.extend(logits.argmax(1).cpu().numpy())
            labels_l.extend(labels.cpu().numpy())
    infer_s = time.time() - t0
    yt, yp = np.array(labels_l), np.array(preds_l)
    acc = accuracy_score(yt, yp)
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    cr = classification_report(yt, yp, target_names=SEVERITY_LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": acc, "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
    }, cr, cm, yt, yp, infer_s


def main():
    cache = torch.load(CACHE_PATH, weights_only=False)
    img_all, txt_all, labels_all = cache["img_feats"], cache["txt_feats"], cache["labels"]
    test_idx = cache["test_idx"]
    print(f"Test samples: {len(test_idx)}")

    model = GatedFusionWide(img_dim=1280, txt_dim=768, num_classes=NUM_CLASSES,
                            dropout=0.3, gate_dropout=0.1).to(DEV)
    model.load_state_dict(torch.load(OUT_DIR / f"{RUN_NAME}_best.pth", map_location=DEV))
    model.eval()
    tr, tt = count_parameters(model)
    print(f"Params: {tr:,} trainable / {tt:,} total")

    # latency: steady-state mean of 3 trials over full test set (fusion head only)
    test_ds = torch.utils.data.TensorDataset(
        img_all[test_idx], txt_all[test_idx], labels_all[test_idx])
    loader = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False)
    lat_ms = []
    with torch.no_grad():
        for _ in range(3):
            t0 = time.time()
            for img_f, txt_f, _ in loader:
                model(img_f.to(DEV), txt_f.to(DEV))
            lat_ms.append((time.time() - t0) / len(loader) * 1000.0)
    fusion_ms = float(np.mean(lat_ms))
    print(f"Fusion-head latency: trials={[round(x, 2) for x in lat_ms]} mean={fusion_ms:.2f} ms/sample-batch64")

    tm, tcr, tcm, yt, yp, infer_s = eval_full(model, loader)

    summary = {"name": RUN_NAME}
    summary.update({k: round(float(v), 4) for k, v in tm.items()})
    for lbl in range(3):
        dn = SEVERITY_LABELS[lbl]
        sn = LABEL_NAMES[lbl]
        summary[f"precision_{sn}"] = round(tcr[dn]["precision"], 4)
        summary[f"recall_{sn}"] = round(tcr[dn]["recall"], 4)
        summary[f"f1_{sn}"] = round(tcr[dn]["f1-score"], 4)

    summary["test_samples"] = len(test_idx)
    summary["confusion_matrix"] = tcm.astype(int).tolist()
    summary["latency"] = {
        "fusion_head_ms_mean": round(fusion_ms, 2),
        "fusion_head_ms_trials": [round(x, 2) for x in lat_ms],
        "image_encoder_ms_unchanged": 92.62,
        "text_encoder_ms_unchanged": 140.19,
        "total_pipeline_ms_est": round(92.62 + 140.19 + fusion_ms, 2),
        "note": "Encoders are the same frozen checkpoints as the original v3 model; "
                "encoder latency is carried over from FINAL_RESULTS.md (measured via "
                "FastAPI /predict on the same machine). Fusion-head latency measured "
                "here on CPU (batch 64, steady-state mean of 3 trials).",
    }

    with open(OUT_DIR / f"{RUN_NAME}_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(OUT_DIR / f"{RUN_NAME}_preds.npz", yt=yt, yp=yp)
    np.save(OUT_DIR / f"{RUN_NAME}_cm.npy", tcm)

    print(f"\n  Acc={tm['accuracy']:.4f} F1m={tm['f1_macro']:.4f} F1w={tm['f1_weighted']:.4f}")
    for lbl in range(3):
        dn = SEVERITY_LABELS[lbl]
        print(f"    {dn:25s}  P={tcr[dn]['precision']:.4f}  R={tcr[dn]['recall']:.4f}  F1={tcr[dn]['f1-score']:.4f}")
    print("  Confusion matrix:\n", tcm)
    print(f"\n  Saved eval JSON -> {OUT_DIR / f'{RUN_NAME}_eval.json'}")


if __name__ == "__main__":
    main()
