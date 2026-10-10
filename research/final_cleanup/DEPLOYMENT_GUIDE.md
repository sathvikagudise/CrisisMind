# CrisisMind — Deployment Guide (verified)

**Purpose:** Reproduce the running CrisisMind demo exactly as verified on 2026-09-01. Every command below was executed successfully in this workspace (Windows, CPU-only, no CUDA).

---

## 1. Topology

```
Browser ──► Vite dev server 127.0.0.1:5173  ──proxy──►  FastAPI uvicorn 127.0.0.1:8000
                                     │ (default VITE_API_URL http://localhost:8000)
                                    axios /predict /health /disasters/events /batch/upload...
```

## 2. Backend (FastAPI, port 8000)

```bash
cd C:\Users\sathv\Downloads\Creations\CrisisMind
backend\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

- Health probe: `GET http://127.0.0.1:8000/health` → `{"status":"healthy","models_loaded":5}`
- Model loading (at startup, from `backend/outputs/checkpoints/`): image backbone (`best_model.pth`), text backbone (`best_text_model.pth`), gated fusion (`best_gated_fusion_v3.pth`), concat fusion (`best_concat_fusion.pth`), plus classifier → **5 components**.
- Interactive API: `GET /docs` (Swagger).

> **Deployment vs final-report checkpoints:** the running demo serves the deployed `best_gated_fusion_v3.pth` head, which is NOT the leak-free `F4_generic_wce_bal_n0.0_s42` head used for final test metrics (69.50%). For a demo that matches the paper numbers exactly, swap the gated checkpoint path in `backend/api/config.py` (GATED_CKPT) to
> `backend/outputs/checkpoints/final_exp/F4_generic_wce_bal_n0.0_s42/best.pth`.
> Config differences: deployed FUSION_DROPOUT=0.3 vs final 0.2; inputs IMG_SIZE 224, TEXT_MAX_LEN 64.

### Verified smoke test (multipart)
```
POST /predict
  image=<file> & tweet_text=<string>
→ {"success":true,"prediction":{"severity":"Severe Damage","confidence":98.23},
   "model":"GatedFusion v3","processing_time_ms":153.85,...}
```
- Note: severity/confidence are nested under `prediction` (top-level `confidence` raises KeyError).
- Error behavior: 400 missing field, 413 >10 MB, 422 unsupported format, 500 inference failure.

### Full endpoint list (29 routes via openapi.json)
Main: `/health`, `/docs`, `/predict`, `/disasters/events`, `/explainability/*` (Grad-CAM), `/batch/upload` → `/batch/download/*`, `/dashboard/*` (aggregates), `/report/*` (wall-clock + PDF), `/analysis-center/*`, `/test-connections/*`, `/dataset-info`.

## 3. Frontend (Vite + React 19, port 5173)

```bash
cd C:\Users\sathv\Downloads\Creations\CrisisMind\frontend
npm install
npm run dev
```

- Dev URL: `http://127.0.0.1:5173` (returns 200 when up).
- Proxies configured in `frontend/vite.config.ts`: `/predict`, `/health`, `/disasters/events`, `/batch/upload`…`/batch/download`, `/dashboard/*`, `/report` → `http://127.0.0.1:8000`.
- API base: `VITE_API_URL || http://localhost:8000`, timeout 120000 ms (`frontend/src/services/api.ts`).
- Production build: `npm run build` (tsc -b && vite build → `dist/`), preview: `npm run preview`.

## 4. Windows process management (verified in this session)

- Backend as a detached background process (survives shell close):
  ```python
  python -c "import subprocess,win32api; r=subprocess.Popen([r'backend\.venv\Scripts\python.exe','-m','uvicorn','backend.api.main:app','--host','127.0.0.1','--port','8000'],cwd=r'C:\Users\sathv\Downloads\Creations\CrisisMind',creationflags=0x08000000); open(r'C:\Users\sathv\Downloads\Creations\CrisisMind\backend.log','w').close()"
  ```
- Kill a lingering port (netstat/taskkill not on PATH here → use python):
  ```python
  python -c "import subprocess; out=subprocess.run(['tasklist','/FI','PID eq <PID>'],capture_output=True,text=True).stdout; print(out)"
  python -c "import subprocess; subprocess.run(['taskkill','/F','/PID',str(<PID>)])"
  ```
- Frontend `npm run dev` run the same way (npm.cmd) — verify via `http://127.0.0.1:5173`.

## 5. Dependencies (pin reference)

- Backend: `backend/requirements.txt` (16 packages: fastapi, uvicorn[standard], python-multipart, torch>=2.0, torchvision, transformers>=4.30, Pillow, pydantic v2, httpx, scikit-learn, matplotlib, seaborn, pandas, tqdm, numpy).
- Frontend: `frontend/package.json` (react 19, vite 8, tailwind v4, shadcn-style radix primitives, axios, framer-motion, recharts, lucide-react).

## 6. Cold start checklist

1. Ports 8000/5173 free (`netstat` via python if needed).
2. Backend venv present (`backend\.venv\Scripts\python.exe -c "import torch"`).
3. Checkpoints exist: `backend/outputs/checkpoints/{best_model,best_text_model,best_gated_fusion_v3,best_concat_fusion}.pth` (deployed) — final head `final_exp/F4_generic_wce_bal_n0.0_s42/best.pth`.
4. Start backend → wait for `/health` `models_loaded:5`.
5. Start frontend → confirm :5173 200.
6. Run `/predict` smoke test; check `/dashboard/*` and Grad-CAM endpoint manually in browser.

## 7. Known deployment caveats for the report/PPT

- CPU-only inference: full pipeline ~133 ms (FINAL_TEST_RESULTS), deployed v3 head slower (~154 ms smoke test) — real-time-capable.
- The deployed head and the reported head differ (see §2 note) — decide before the demo which numbers to quote; do not mix.
- `research/assets/screenshots/` empty — capture screenshots for the PPT before submission (`report_data_package.md` §U).

---
*Verified live 2026-09-01 on Windows (AMD64, Python 3.11 CPU).*