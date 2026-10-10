# 01 — Project Structure Audit

## Overview

Full project tree reconstructed from the workspace root (`CrisisMind/`). All paths relative to root.

## Complete Tree

```
CrisisMind/
├── api_server.log                          # Runtime log (generated during testing)
├── output.log                              # Runtime log (generated during training)
├── training_output.log                     # Runtime log (generated during training)
│
├── backend/                                # BACKEND PACKAGE
│   ├── requirements.txt                    # Python dependencies (8 packages)
│   ├── README.md                           # MISSING
│   │
│   ├── api/                                # FastAPI service layer
│   │   ├── __init__.py                     # Empty marker
│   │   ├── config.py                       # API config (checkpoint paths, device, model params)
│   │   ├── load_model.py                   # Model loading at startup
│   │   ├── main.py                         # FastAPI app, routes, CORS, validation
│   │   ├── predict.py                      # Prediction pipeline
│   │   ├── preprocessing.py                # Image transforms + text tokenization
│   │   ├── schemas.py                      # Pydantic models
│   │   └── README.md                       # API documentation (152 lines)
│   │
│   ├── datasets/                           # Data loading & preprocessing
│   │   ├── __init__.py                     # MISSING
│   │   ├── analyze_severity_text.py        # EDA script (not a dataset loader)
│   │   ├── data_loader.py                  # Image dataset loader
│   │   ├── multimodal_data.py              # Multimodal data loading
│   │   ├── multimodal_dataset.py           # Multimodal PyTorch Dataset
│   │   ├── prepare_data.py                 # TSV loading + label mapping
│   │   ├── text_dataset.py                 # Text PyTorch Dataset
│   │   └── text_preprocessing.py           # Text cleaning utility (4-line function)
│   │
│   ├── models/                             # Model architectures
│   │   ├── __init__.py                     # MISSING
│   │   ├── efficientnet.py                 # EfficientNet-B0 with classifier swap
│   │   ├── fusion_models.py                # ImageFeatureExtractor, TextFeatureExtractor, ConcatFusion, GatedFusion
│   │   └── text_model.py                   # DistilBERT text classifier
│   │
│   ├── outputs/                            # Generated outputs (not committed to git)
│   │   ├── cached_features/                # Pre-extracted image+text features
│   │   │   ├── train/features.pt           #   20 MB (2468 samples)
│   │   │   ├── val/features.pt            #  4.2 MB (529 samples)
│   │   │   └── test/features.pt           #  4.2 MB (529 samples)
│   │   ├── checkpoints/                    # Trained model weights
│   │   │   ├── best_model.pth              #   16 MB — Image Baseline
│   │   │   ├── best_text_model.pth         #  256 MB — Text Baseline
│   │   │   ├── best_concat_fusion.pth      #  4.1 MB — Concat Fusion
│   │   │   └── best_gated_fusion.pth       #  4.6 MB — Gated Fusion
│   │   ├── metrics/                        # Evaluation CSVs
│   │   │   ├── classification_report.csv   # Image baseline
│   │   │   ├── test_metrics.csv            # Image baseline
│   │   │   ├── text_classification_report.csv
│   │   │   ├── text_metrics.csv
│   │   │   ├── concat_fusion_metrics.csv
│   │   │   └── gated_fusion_metrics.csv
│   │   ├── plots/                          # 13 PNG files
│   │   │   ├── class_distribution_*.png    # 3 files
│   │   │   ├── confusion_matrix_*.png      # 4 files
│   │   │   ├── training_curves_*.png       # 3 files (image, concat, gated)
│   │   │   ├── fusion_gate_distribution.png
│   │   │   └── fusion_modality_weights.png
│   │   └── reports/                        # Markdown reports
│   │       ├── evaluation_summary.md
│   │       ├── fusion_comparison_report.md
│   │       └── text_evaluation_summary.md
│   │
│   ├── services/                           # EMPTY — placeholder directory
│   ├── utils/                              # EMPTY — placeholder directory
│   │
│   └── training/                           # Training & evaluation scripts
│       ├── config.py                       # Training config (paths, hyperparams)
│       ├── trainer.py                      # Image baseline training
│       ├── text_trainer.py                 # Text baseline training
│       ├── fusion_trainer.py               # Fusion model training
│       ├── fusion_comparison.py            # Fusion model comparison
│       ├── fusion_evaluate.py              # Fusion model evaluation
│       ├── evaluate.py                     # Image baseline evaluation
│       ├── text_evaluate.py                # Text baseline evaluation
│       └── visualize.py                    # Plotting utilities
│
├── frontend/                               # REACT FRONTEND
│   ├── index.html                          # Vite entry HTML
│   ├── package.json                        # Node dependencies (16 runtime + 7 dev)
│   ├── package-lock.json                   # Lockfile
│   ├── vite.config.ts                      # Vite config (proxy, Tailwind, path alias)
│   ├── tsconfig.json                       # TypeScript config
│   ├── tsconfig.app.json                   # App-specific TS config
│   ├── tsconfig.node.json                  # Node-specific TS config
│   ├── .gitignore
│   ├── .oxlintrc.json                      # Linter config
│   ├── README.md                           # Setup instructions
│   ├── public/
│   │   └── favicon.svg                     # Favicon
│   │
│   └── src/
│       ├── main.tsx                        # React entry point
│       ├── App.tsx                         # Router (3 routes)
│       ├── index.css                       # Tailwind v4 + custom theme
│       ├── vite-env.d.ts                   # Vite client types
│       ├── css.d.ts                        # ❌ DEAD FILE (redundant with vite-env.d.ts)
│       │
│       ├── components/
│       │   ├── Layout.tsx                  # Page layout wrapper (Navbar + Footer + motion)
│       │   ├── Navbar.tsx                  # Sticky nav, mobile hamburger, GitHub link
│       │   ├── Footer.tsx                  # Grid footer with version badge
│       │   ├── FeatureCard.tsx             # Reusable feature card with hover animation
│       │   ├── ComingSoonCard.tsx           # Disabled card with lock icon
│       │   ├── UploadCard.tsx              # Drag & drop image upload with preview
│       │   ├── PredictionCard.tsx          # Prediction result display
│       │   ├── LoadingSpinner.tsx          # Animated spinner with sequential messages
│       │   ├── ErrorAlert.tsx              # Animated error/warning alert
│       │   ├── AnimatedCounter.tsx         # Scroll-triggered number animation
│       │   ├── Timeline.tsx                # Horizontal research roadmap
│       │   ├── SystemStatus.tsx             # Live health polling widget
│       │   └── ui/                         # Primitive components
│       │       ├── button.tsx
│       │       ├── card.tsx
│       │       ├── badge.tsx
│       │       └── skeleton.tsx            # ❌ DEFINED BUT NEVER USED
│       │
│       ├── pages/
│       │   ├── Home/index.tsx              # Landing page (7 sections)
│       │   ├── Analysis/index.tsx          # Analysis dashboard (step workflow)
│       │   └── About/index.tsx             # About page (card grid)
│       │
│       ├── services/
│       │   └── api.ts                      # Axios client (2 endpoints)
│       │
│       ├── types/
│       │   └── index.ts                    # TypeScript interfaces
│       │
│       ├── lib/
│       │   └── utils.ts                    # cn() helper (clsx + tailwind-merge)
│       │
│       ├── hooks/                          # EMPTY
│       ├── assets/                         # EMPTY
│       └── styles/                         # EMPTY
│
├── research/                               # RESEARCH DOCUMENTATION
│   ├── README.md                           # Research overview
│   ├── 00_Project_Overview.md
│   ├── 01_Dataset_Analysis.md
│   ├── 02_Image_Baseline.md
│   ├── 03_Text_Baseline.md
│   ├── 04_Multimodal_Fusion.md
│   ├── 05_Experimental_Results.md
│   ├── 06_Project_Architecture.md          # Mermaid diagrams
│   ├── 07_Research_Log.md
│   ├── 08_Novelty_and_Contribution.md
│   ├── assets/                             # ❌ EMPTY — all image references broken
│   └── references/
│       ├── README.md                       # 7 citations
│       ├── dataset_summary_report.md
│       ├── evaluation_summary.md
│       ├── fusion_comparison_report.md
│       └── text_evaluation_summary.md
│
├── project_audit/                          # THIS AUDIT (newly created)
│   └── (10 markdown files)
│
└── root scripts
    ├── dataset_audit.py                    # Dataset audit script (266 lines)
    ├── dataset_summary_report.md           # ❌ DUPLICATE (copy in research/references/)
    ├── run_fusion.py                       # Fusion training orchestration (318 lines)
    ├── run_image_baseline.py               # Image baseline training (149 lines)
    ├── run_text_baseline.py                # Text baseline training (182 lines)
    └── run_text_eval.py                    # Text baseline evaluation (144 lines)
```

## Summary

| Metric | Count |
|---|---|
| Total Python files | 27 |
| Total TypeScript/TSX files | 24 |
| Total Markdown files | 20 |
| Total directories | 28 |

## Issues Found

### Empty / Placeholder Directories
| Path | Status |
|---|---|
| `backend/services/` | ❌ Empty |
| `backend/utils/` | ❌ Empty |
| `frontend/src/hooks/` | ❌ Empty |
| `frontend/src/assets/` | ❌ Empty |
| `frontend/src/styles/` | ❌ Empty |
| `research/assets/` | ❌ Empty (but 13 plot PNGs exist in `backend/outputs/plots/`) |

### Dead Files
| File | Issue |
|---|---|
| `frontend/src/css.d.ts` | Redundant with `vite-env.d.ts` |
| `frontend/src/components/ui/skeleton.tsx` | Defined but never imported anywhere |

### Duplicate Files
| File 1 | File 2 | Content |
|---|---|---|
| `dataset_summary_report.md` (root) | `research/references/dataset_summary_report.md` | Identical |
| `backend/outputs/reports/evaluation_summary.md` | `research/references/evaluation_summary.md` | Identical |
| `backend/outputs/reports/fusion_comparison_report.md` | `research/references/fusion_comparison_report.md` | Identical |
| `backend/outputs/reports/text_evaluation_summary.md` | `research/references/text_evaluation_summary.md` | Identical |

### Missing Files
| Expected | Location | Impact |
|---|---|---|
| `backend/README.md` | `backend/` | No backend-level README |
| `__init__.py` | `backend/datasets/` | Package not importable without it |
| `__init__.py` | `backend/models/` | Package not importable without it |
| `text_baseline_curves.png` | `backend/outputs/plots/` | Documented as not preserved |
| `Dockerfile` | Root or `backend/` | Referenced in `backend/api/README.md` but doesn't exist |
| Architecture diagrams (PNG) | `research/assets/` | Only Mermaid code blocks exist |
