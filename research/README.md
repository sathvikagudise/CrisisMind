# CrisisMind — Research Knowledge Base

## Project Overview

CrisisMind is a multimodal deep learning framework for automated disaster damage severity assessment using social media data from Twitter. The system processes paired image-text content from crisis-related tweets and classifies damage severity into three levels: **little or no damage**, **mild damage**, and **severe damage**.

## Completed Phases

| Phase | Status | Description |
|---|---|---|
| Phase 0 — Dataset Audit | ✅ Complete | CrisisMMD v2.0 dataset exploration and statistical analysis |
| Phase 1 — Image Baseline | ✅ Complete | EfficientNet-B0 pretrained model for image-only severity classification |
| Phase 2 — Text Baseline | ✅ Complete | DistilBERT transformer model for text-only severity classification |
| Phase 3 — Multimodal Fusion | ✅ Complete | Concatenation and Gated Fusion combining image + text modalities |
| Phase 4 — Disaster Impact Index | 🔜 Planned | Weighted severity scoring framework |
| Phase 5 — Explainable AI | 🔜 Planned | Grad-CAM and attention visualization |
| Phase 6 — Deployment | 🔜 Planned | FastAPI backend + React dashboard |

## Repository Structure

```
CrisisMind/
├── dataset_analysis/
│   ├── CrisisMMD_v2.0/          # Dataset
│   │   ├── annotations/          # 7 TSV annotation files
│   │   ├── data_image/           # Disaster images (7 events)
│   │   └── json/                 # Raw Twitter JSON data
│   └── reports/                  # Dataset analysis charts
├── backend/
│   ├── datasets/                 # Data loading and preprocessing
│   ├── models/                   # Model architectures
│   ├── training/                 # Training and evaluation
│   └── outputs/                  # All artifacts
│       ├── checkpoints/          # Trained model weights
│       ├── metrics/              # CSV evaluation metrics
│       ├── plots/                # Training curves and CM
│       └── reports/              # Evaluation summaries
├── research/                     # Documentation (this folder)
├── dataset_audit.py              # Dataset analysis script
├── run_image_baseline.py         # Phase 1 runner
├── run_text_baseline.py          # Phase 2 runner
└── run_fusion.py                 # Phase 3 runner
```

## Results Summary

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Image Baseline (EfficientNet-B0) | 0.5841 | 0.4869 | 0.5959 |
| Text Baseline (DistilBERT) | 0.5250 | 0.4221 | 0.5328 |
| Concat Fusion | 0.6371 | 0.5302 | 0.6400 |
| **Gated Fusion (Best)** | **0.6352** | **0.5375** | **0.6415** |

The proposed **Gated Fusion** model achieves the highest Macro F1 score, demonstrating the effectiveness of learnable modality weighting for disaster damage assessment.

## Future Roadmap

1. **DII Framework**: Normalize predictions into a continuous Disaster Impact Index (0–100)
2. **Explainable AI**: Integrate Grad-CAM for images and attention visualization for text
3. **Decision Intelligence**: Risk scoring and confidence-based decision support
4. **Deployment**: FastAPI backend with React dashboard for real-time inference
