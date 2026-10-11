# CrisisMind

A multimodal deep learning platform that performs real-time disaster damage severity assessment from social-media (image + text) content.

CrisisMind pairs crisis tweet **images** with their **text**, evaluates them against a three-level damage-severity taxonomy, and produces explainable, decision-ready outputs (severity, confidence, priority, and recommended actions). The system is served through a FastAPI backend and a React dashboard.

---

## Overview

CrisisMind is a research-grade, **leakage-free** multimodal classification system for automated disaster damage severity assessment. It combines a frozen **EfficientNet-B0** image encoder with a frozen **DistilBERT** text encoder through an **Adaptive Gated Fusion** head, selecting the winning configuration on **validation macro-F1 only** and evaluating on the held-out test set exactly once.

The deployed model (`F4_generic_wce_bal_n0.0_s42`) was chosen after finding — and correcting — **70.6% inherited encoder leakage** in earlier splits; the final result is therefore trustworthy by construction. See `research/final_experiment/` and `research/final_cleanup/` for the full protocol and audit trail.

**Severity classes:** `little_or_no_damage`, `mild_damage`, `severe_damage`.

---

## Features

- **Image-only classification** — EfficientNet-B0 severity head
- **Text-only classification** — DistilBERT severity head
- **Multimodal Adaptive Gated Fusion (best)** — learnable per-modality weighting
- **Leakage-free evaluation protocol** — validation-macro-F1 selection, single test pass
- **Automatic analysis-mode routing** — image-only, text-only, or multimodal
- **Batch processing** — upload many images + an optional CSV, run, and export
- **Grad-CAM explainability** — visual heatmaps overlaid on the input image
- **Live disaster intelligence** — event ingestion from NASA EONET, GDACS, and ReliefWeb
- **Analytics dashboard** — summary stats, charts, and a timeline of analyses
- **Analysis Center** — searchable, filterable, exportable record store
- **Report generation** — PDF and CSV reports for manual, live, and batch analyses
- **Context-aware decision support** — priority, emergency level, infrastructure impact, and recommended actions

---

## Tech Stack

**Frontend**
- React 19 + TypeScript
- Vite 8
- TailwindCSS 4
- React Router DOM 7
- Axios
- Recharts
- Framer Motion, Lucide React, Radix UI

**Backend**
- Python 3.11
- FastAPI + Uvicorn
- PyTorch (CPU) + torchvision
- Hugging Face Transformers (DistilBERT)
- Pillow, NumPy, pandas, scikit-learn
- Matplotlib, seaborn
- HTTPX (async external providers)
- fpdf2 (PDF compiler)

**Models**
- EfficientNet-B0 (frozen ImageNet encoder + classifier head)
- DistilBERT (frozen encoder + classifier head)
- Adaptive Gated Fusion v3 (`GatedFusionWide`)

**Storage**
- In-memory stores (stateless API; no external database).

---

## Folder Structure

```
CrisisMind/
│
├── backend/
│   ├── api/                     # FastAPI app, routes, schemas, preprocessing
│   ├── analysis_engine/         # Orchestration: model loading, selection, decisioning
│   ├── models/                  # EfficientNet-B0, DistilBERT, Gated Fusion
│   ├── datasets/                # Data loading and preprocessing
│   ├── training/                # Training / evaluation scripts
│   ├── batch_processing/        # Batch upload → run → export
│   ├── dashboard/               # Analytics store + service
│   ├── disaster_intelligence/   # NASA EONET / GDACS / ReliefWeb providers
│   ├── explainability/          # Grad-CAM + overlays
│   ├── reporting/               # PDF + CSV report builders
│   ├── analysis_center/         # Search / filter / export records
│   ├── config/                  # Shared constants + checkpoint paths
│   ├── tests/                   # Backend tests
│   ├── Dockerfile
│   ├── requirements.txt
│   └── outputs/                 # Artifacts (checkpoints, metrics, plots) — gitignored
│
├── frontend/
│   ├── src/
│   │   ├── pages/               # Home, Analysis, Disasters, Batch, Dashboard, AnalysisCenter, About
│   │   ├── components/          # UI components (cards, navbar, charts, ...)
│   │   ├── services/api.ts      # Axios client (uses VITE_API_URL)
│   │   ├── types/
│   │   └── lib/
│   ├── vite.config.ts
│   ├── vercel.json
│   └── package.json
│
├── research/                    # Paper assets, final experiment, data cleaning, audits
├── dataset_analysis/            # CrisisMMD v2.0 dataset — gitignored (large)
├── project_audit/               # Structural / model / security audits
├── testing/                     # Validation reports + evaluation scripts
├── render.yaml                  # Render blueprint (backend)
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/sathvikagudise/CrisisMind.git
cd CrisisMind
```

### 2. Set up the backend

```bash
python -m venv backend/.venv
# Windows:
backend/.venv/Scripts/activate
# macOS / Linux:
source backend/.venv/bin/activate

pip install -r backend/requirements.txt
```

### 3. Run the backend

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

The first startup loads five models and may take 10–40 s on CPU. Verify:

```bash
curl http://localhost:8000/health
# {"status":"healthy","models_loaded":5}
```

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. During local development, Vite proxies API calls to `http://localhost:8000`.

---

## Model Weights

The three deployed checkpoints are required at runtime:

| Checkpoint | Path | Size |
|---|---|---|
| Image classifier | `backend/outputs/checkpoints/best_model.pth` | ~16 MB |
| Text classifier | `backend/outputs/checkpoints/best_text_model.pth` | ~255 MB |
| Gated fusion (final) | `backend/outputs/checkpoints/final_exp/F4_generic_wce_bal_n0.0_s42/best.pth` | ~12 MB |

They are distributed via **Git LFS** (install `git-lfs`, then `git lfs pull`). The frozen generic encoders (ImageNet EfficientNet-B0, vanilla DistilBERT) are downloaded automatically by `torchvision`/`transformers` on first run.

---

## Environment Variables

Create a `.env` where needed, or set these in your hosting dashboard.

| Variable | Where | Description |
|---|---|---|
| `VITE_API_URL` | Frontend | Base URL of the backend API (defaults to `http://localhost:8000`) |
| `PORT` | Backend | Port to bind (Render sets this automatically) |

The core system needs **no secrets** — disaster providers (NASA EONET, GDACS, ReliefWeb) are keyless and CORS is configured to allow the deployment origins.

---

## Deployment

- **Frontend → Vercel** (static SPA). `frontend/vercel.json` sets the build (`npm run build`), output (`dist`), and SPA rewrite.
  Set `VITE_API_URL` to your deployed backend URL.
- **Backend → Render** (Docker web service). `render.yaml` + `backend/Dockerfile` build a CPU-only image, pull the LFS checkpoints, and serve `backend.api.main:app` via Uvicorn.

```bash
# Render start command (equivalent)
uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
```

> The backend needs ~2 GB RAM (DistilBERT + EfficientNet + PyTorch).

---

## Testing

Run the backend test suite:

```bash
pytest
```

Evaluation and validation scripts live in `testing/` (e.g. `full_evaluation.py`, `fast_eval.py`, `diagnose_predictions.py`).

---

## API Documentation

Interactive docs are served at `/docs` (Swagger) and `/redoc`. Example endpoint:

**`POST /predict`**

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@sample.jpg" \
  -F "tweet_text=Homes destroyed by the wildfire"
```

Response (abbreviated):

```json
{
  "analysis_id": "…",
  "success": true,
  "analysis_mode": "MULTIMODAL",
  "prediction": { "severity": "Severe Damage", "confidence": 0.99 },
  "decision": { "priority": "HIGH", "emergency_level": "…", "recommended_actions": ["…"] },
  "processing_time_ms": 133.4
}
```

**Core routes**

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness + models loaded |
| POST | `/predict` | Image and/or text severity prediction |
| GET | `/disasters/events` | Live disaster events |
| GET | `/disasters/events/{id}` | Event detail |
| POST | `/disasters/events/{id}/analyze` | Analyze an event |
| POST | `/batch/upload` | Upload a batch |
| POST | `/batch/start/{batch_id}` | Start batch processing |
| GET | `/batch/status/{batch_id}` | Batch status |
| GET | `/batch/results/{batch_id}` | Batch results |
| GET | `/batch/download/csv/{batch_id}` | Download batch CSV |
| GET | `/batch/download/pdf/{batch_id}` | Download batch PDF |
| GET | `/dashboard/summary` `/statistics` `/charts` `/timeline` `/recent` | Analytics |
| GET | `/report/manual/{id}` `/live/{id}` `/batch/{id}` `/csv/{id}` | Reports |
| POST | `/explainability/gradcam` | Grad-CAM explanation |
| GET | `/analysis-center` (+ `/search` `/export` `/statistics` `/{id}`) | Analysis Center |

---

## Results

Final model: **Adaptive Gated Fusion v3 (`F4_generic_wce_bal_n0.0_s42`)**, frozen generic encoders, leakage-free.

| Metric | Value |
|---|---|
| Test Accuracy | **0.6950** |
| Macro Precision | 0.5958 |
| Macro Recall | 0.5480 |
| Macro F1 | **0.5650** |
| Weighted F1 | 0.6794 |
| Test samples | 436 |

**Ablation (test accuracy):**

| Model | Test Acc | Macro F1 |
|---|---|---|
| Image-only (EfficientNet-B0) | 0.6468 | 0.5335 |
| Text-only (DistilBERT) | 0.5367 | 0.4625 |
| Concatenation fusion | 0.6468 | 0.5618 |
| **Adaptive Gated Fusion v3 (final)** | **0.6950** | **0.5650** |

**Latency (CPU, batch size 1):** full pipeline ~133 ms; fusion head ~1.35 ms.

**Dataset:** CrisisMMD v2.0 — 18,082 annotations across 7 disaster events; 3,526 valid image + damage-severity labels, cleaned to 3,105 (train 2,521 / val 460 / test 436).

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Added new feature"`
4. Push: `git push origin feature-name`
5. Open a Pull Request

---

## Security

- No secrets committed; deployment uses environment variables
- CORS middleware configures allowed origins
- Stateless, in-memory API architecture
- Input validation (image type + size limits) on the prediction endpoint

---

## License

This project is licensed under the MIT License.

---

## Author

**Gudise Sathvika**

- GitHub: https://github.com/sathvikagudise
- LinkedIn: https://linkedin.com/in/sathvikayadav
- Email: sathvikayadav3@gmail.com

---

## Support

If you like this project, give it a star on GitHub.
