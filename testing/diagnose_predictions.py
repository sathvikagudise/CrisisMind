"""Diagnostic script: run all 3 models on test images + text, dump raw logits."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import csv
import time
from pathlib import Path
from PIL import Image

from backend.analysis_engine.model_loader import load_all_models
from backend.api.preprocessing import preprocess_image, tokenize_text
from backend.api.config import SEVERITY_MAP, DEVICE
from backend.config.shared import GATED_CKPT, IMG_CKPT, TEXT_CKPT

LABEL_NAMES = {0: "little_or_no_damage", 1: "mild_damage", 2: "severe_damage"}

# Ground-truth labels from the test images (filename suffix: _0=little, _1=severe, no suffix=unknown)
TEST_DIR_IMG = Path(__file__).parent / "manual_test"
TEST_DIR_SAMPLE = Path(__file__).parent / "sample_images"

# Text samples with known labels
TEXT_SAMPLES = [
    ("Little or No Damage", "The road has a few visible cracks but traffic is normal."),
    ("Little or No Damage", "Some tree branches fell on the lawn but the house is intact."),
    ("Little or No Damage", "Light rain caused minor puddles on the street."),
    ("Little or No Damage", "A few roof tiles shifted slightly during the storm."),
    ("Little or No Damage", "The fence has some minor scratches from wind-blown debris."),
    ("Mild Damage", "Several buildings have broken windows and partial wall damage."),
    ("Mild Damage", "Floodwater has entered the ground floor of the apartment complex."),
    ("Mild Damage", "The church roof collapsed partially due to heavy snow load."),
    ("Mild Damage", "Moderate cracks in the retaining wall along the highway."),
    ("Mild Damage", "Three vehicles damaged by falling tree branches after the storm."),
    ("Severe Damage", "Multiple buildings have collapsed and rescue teams are evacuating people."),
    ("Severe Damage", "Entire neighbourhood destroyed by the wildfire leaving only ashes."),
    ("Severe Damage", "The hospital has severe structural damage and is being evacuated."),
    ("Severe Damage", "Hundreds of homes washed away by the catastrophic flood waters."),
    ("Severe Damage", "Major earthquake has reduced the city center to complete rubble."),
]


def infer_filename_label(fname):
    """Infer ground truth from CrisisMMD filename convention."""
    stem = Path(fname).stem
    parts = stem.split("_")
    if parts[-1] == "0":
        return "little_or_no_damage"
    elif parts[-1] == "1":
        return "severe_damage"
    return "unknown"


def run():
    print("=" * 80)
    print("  CrisisMind Inference Diagnostic")
    print("=" * 80)

    # Load all models
    print("\nLoading models...")
    t0 = time.time()
    models = load_all_models()
    print(f"Models loaded in {time.time()-t0:.1f}s on {DEVICE}")

    # Verify checkpoint loading
    print(f"\nCheckpoint sizes:")
    for name, path in [("IMG_CKPT", IMG_CKPT), ("TEXT_CKPT", TEXT_CKPT), ("GATED_CKPT", GATED_CKPT)]:
        sz = os.path.getsize(path) / 1e6
        print(f"  {name}: {path.name} ({sz:.1f} MB)")

    rows = []

    # ---- TEST 1: Image-only on manual_test images ----
    print("\n" + "-" * 80)
    print("TEST 1: Image-only inference on manual_test/ images")
    print("-" * 80)

    if not TEST_DIR_IMG.exists():
        print(f"  WARNING: {TEST_DIR_IMG} does not exist")
    else:
        for f in sorted(TEST_DIR_IMG.glob("*.jpg")):
            gt = infer_filename_label(f.name)
            img = Image.open(f).convert("RGB")
            tensor = preprocess_image(img).to(DEVICE)

            t0 = time.time()
            with torch.no_grad():
                logits = models["image_classifier"](tensor)
            latency = (time.time() - t0) * 1000

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(torch.argmax(logits, dim=1).item())
            conf = probs[pred] * 100

            print(f"  {f.name:30s}  GT={gt:25s}  Pred={SEVERITY_MAP[pred]:25s}  "
                  f"Conf={conf:5.1f}%  Raw=[{probs[0]:.4f},{probs[1]:.4f},{probs[2]:.4f}]  "
                  f"{latency:.0f}ms")
            rows.append({
                "test": "image_only_manual",
                "filename": f.name,
                "gt_label": gt,
                "predicted": SEVERITY_MAP[pred],
                "pred_idx": pred,
                "confidence": round(conf, 2),
                "p0": round(float(probs[0]), 4),
                "p1": round(float(probs[1]), 4),
                "p2": round(float(probs[2]), 4),
                "raw_logits": str(logits.cpu().numpy().tolist()),
                "latency_ms": round(latency, 1),
            })

    # ---- TEST 2: Multimodal on manual_test images (with dummy text) ----
    print("\n" + "-" * 80)
    print("TEST 2: Multimodal (gated fusion) inference on manual_test/ images + dummy text")
    print("-" * 80)

    dummy_texts = [
        "Disaster damage reported in the area",
        "Buildings collapsed after earthquake",
        "Minor flooding on residential streets",
    ]

    if not TEST_DIR_IMG.exists():
        print(f"  WARNING: {TEST_DIR_IMG} does not exist")
    else:
        for i, f in enumerate(sorted(TEST_DIR_IMG.glob("*.jpg"))[:5]):  # just first 5
            gt = infer_filename_label(f.name)
            img = Image.open(f).convert("RGB")
            tensor = preprocess_image(img).to(DEVICE)
            text = dummy_texts[i % len(dummy_texts)]
            input_ids, attn_mask = tokenize_text(text)
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)

            t0 = time.time()
            with torch.no_grad():
                img_feat = models["img_encoder"](tensor)
                txt_feat = models["txt_encoder"](input_ids, attn_mask)
                logits, gates = models["gated_model"](img_feat, txt_feat)
            latency = (time.time() - t0) * 1000

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(torch.argmax(logits, dim=1).item())
            conf = probs[pred] * 100
            g_img = gates[0, 0].item()
            g_txt = gates[0, 1].item()

            print(f"  {f.name:30s}  GT={gt:25s}  Pred={SEVERITY_MAP[pred]:25s}  "
                  f"Conf={conf:5.1f}%  Gates[img={g_img:.3f},txt={g_txt:.3f}]  "
                  f"Raw=[{probs[0]:.4f},{probs[1]:.4f},{probs[2]:.4f}]  "
                  f"{latency:.0f}ms")
            rows.append({
                "test": "multimodal_manual",
                "filename": f.name,
                "gt_label": gt,
                "predicted": SEVERITY_MAP[pred],
                "pred_idx": pred,
                "confidence": round(conf, 2),
                "p0": round(float(probs[0]), 4),
                "p1": round(float(probs[1]), 4),
                "p2": round(float(probs[2]), 4),
                "raw_logits": str(logits.cpu().numpy().tolist()),
                "latency_ms": round(latency, 1),
                "gate_img": round(g_img, 4),
                "gate_txt": round(g_txt, 4),
                "text": text,
            })

    # ---- TEST 3: Text-only on manual text samples ----
    print("\n" + "-" * 80)
    print("TEST 3: Text-only inference on manual text samples")
    print("-" * 80)

    for gt_label, text in TEXT_SAMPLES:
        input_ids, attn_mask = tokenize_text(text)
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            outputs = models["text_classifier"](input_ids, attn_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
        latency = (time.time() - t0) * 1000

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(torch.argmax(logits, dim=1).item())
        conf = probs[pred] * 100

        match = "OK" if SEVERITY_MAP[pred] == gt_label else "WRONG"
        print(f"  [{match:5s}] GT={gt_label:25s}  Pred={SEVERITY_MAP[pred]:25s}  "
              f"Conf={conf:5.1f}%  Raw=[{probs[0]:.4f},{probs[1]:.4f},{probs[2]:.4f}]  "
              f"{latency:.0f}ms  \"{text[:50]}...\"")
        rows.append({
            "test": "text_only_manual",
            "filename": "",
            "gt_label": gt_label,
            "predicted": SEVERITY_MAP[pred],
            "pred_idx": pred,
            "confidence": round(conf, 2),
            "p0": round(float(probs[0]), 4),
            "p1": round(float(probs[1]), 4),
            "p2": round(float(probs[2]), 4),
            "raw_logits": str(logits.cpu().numpy().tolist()),
            "latency_ms": round(latency, 1),
            "text": text,
        })

    # ---- TEST 4: Text-only on sample_images (no useful text, just see bias) ----
    print("\n" + "-" * 80)
    print("TEST 4: Image-only on sample_images/ (diverse set)")
    print("-" * 80)

    if not TEST_DIR_SAMPLE.exists():
        print(f"  WARNING: {TEST_DIR_SAMPLE} does not exist")
    else:
        counts = {0: 0, 1: 0, 2: 0}
        total = 0
        for f in sorted(TEST_DIR_SAMPLE.glob("*.jpg"))[:22]:
            img = Image.open(f).convert("RGB")
            tensor = preprocess_image(img).to(DEVICE)

            with torch.no_grad():
                logits = models["image_classifier"](tensor)

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(torch.argmax(logits, dim=1).item())
            conf = probs[pred] * 100
            counts[pred] += 1
            total += 1
            print(f"  {f.name:35s}  Pred={SEVERITY_MAP[pred]:25s}  "
                  f"Conf={conf:5.1f}%  Raw=[{probs[0]:.4f},{probs[1]:.4f},{probs[2]:.4f}]")

        print(f"\n  Distribution: Little={counts[0]}/{total}  Mild={counts[1]}/{total}  Severe={counts[2]}/{total}")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    for test_name in ["image_only_manual", "multimodal_manual", "text_only_manual"]:
        subset = [r for r in rows if r["test"] == test_name]
        if not subset:
            continue
        preds = [r["pred_idx"] for r in subset]
        correct = sum(1 for r in subset if r["predicted"] == r["gt_label"])
        total = len(subset)
        dist = {i: preds.count(i) for i in range(3)}
        avg_conf = sum(r["confidence"] for r in subset) / total if total else 0
        print(f"\n  {test_name}:")
        print(f"    Samples: {total}  Correct: {correct}  Accuracy: {correct/total*100:.1f}%")
        print(f"    Distribution: Little={dist[0]}  Mild={dist[1]}  Severe={dist[2]}")
        print(f"    Avg confidence: {avg_conf:.1f}%")

    # ---- Save CSV ----
    csv_path = Path(__file__).parent / "INFERENCE_DEBUG.csv"
    fieldnames = ["test", "filename", "gt_label", "predicted", "pred_idx", "confidence",
                  "p0", "p1", "p2", "raw_logits", "latency_ms",
                  "gate_img", "gate_txt", "text"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved to: {csv_path}")


if __name__ == "__main__":
    run()
