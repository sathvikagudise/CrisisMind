# 07 — Dependency Audit

## Python Dependencies

### From `backend/requirements.txt`

| Package | Version Spec | Installed? | Used In | Purpose |
|---|---|---|---|---|
| `fastapi` | >=0.104.0 | ✅ | api/main.py | Web framework |
| `uvicorn[standard]` | >=0.24.0 | ✅ | Run script | ASGI server |
| `python-multipart` | >=0.0.6 | ✅ | api/main.py | File upload parsing |
| `torch` | >=2.0.0 | ✅ | All model files | Deep learning framework |
| `torchvision` | >=0.15.0 | ✅ | models/efficientnet.py | Image models + transforms |
| `transformers` | >=4.30.0 | ✅ | models/text_model.py | HuggingFace Transformers |
| `Pillow` | >=10.0.0 | ✅ | datasets/data_loader.py | Image loading |
| `pydantic` | >=2.0.0 | ✅ | api/schemas.py | Request/response validation |

### Missing Python Dependencies

| Package | Needed By | Reason |
|---|---|---|
| `scikit-learn` | `training/trainer.py`, `training/evaluate.py` | Used for `f1_score`, `classification_report`, `accuracy_score` — but **NOT LISTED** in requirements.txt |
| `matplotlib` | `training/visualize.py` | Used for all plots — **NOT LISTED** |
| `seaborn` | `training/visualize.py` | Used for confusion matrix heatmap — **NOT LISTED** |
| `pandas` | `datasets/prepare_data.py`, `fusion_comparison.py` | Used for TSV loading, DataFrame operations — **NOT LISTED** |
| `tqdm` | `training/trainer.py`, multiple files | Progress bars — **NOT LISTED** |
| `numpy` | Multiple files | Array operations — **NOT LISTED** |
| `logging` | api/main.py | Stdlib — not in requirements ✅ |

**Verdict**: `requirements.txt` lists only 8 packages but the codebase imports at least 6 more (`sklearn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`, `numpy`). This is a **documentation gap** — these packages will be installed as transitive dependencies or are already present, but a fresh install would fail.

### Python Dependency Install Test

```
pip install -r backend/requirements.txt
# MISSING: scikit-learn, matplotlib, seaborn, pandas, tqdm, numpy
# Install commands will fail if these are not transitive deps
```

## Node Dependencies

### From `frontend/package.json`

#### Runtime Dependencies (16 listed)

| Package | Version | Used? | Verified |
|---|---|---|---|
| `@radix-ui/react-dialog` | ^1.1.17 | ❌ **UNUSED** | Never imported |
| `@radix-ui/react-slot` | ^1.3.0 | ❌ **UNUSED** | Never imported |
| `@radix-ui/react-toast` | ^1.2.17 | ❌ **UNUSED** | Never imported |
| `@tailwindcss/vite` | ^4.3.1 | ✅ | vite.config.ts |
| `axios` | ^1.18.1 | ✅ | services/api.ts |
| `class-variance-authority` | ^0.7.1 | ❌ **UNUSED** | Button implements variants manually |
| `clsx` | ^2.1.1 | ✅ | lib/utils.ts |
| `framer-motion` | ^12.42.0 | ✅ | All components |
| `lucide-react` | ^1.21.0 | ✅ | Icons throughout |
| `react` | ^19.2.7 | ✅ | Core |
| `react-dom` | ^19.2.7 | ✅ | Core |
| `react-hook-form` | ^7.80.0 | ❌ **UNUSED** | Never imported |
| `react-router-dom` | ^7.18.0 | ✅ | App.tsx |
| `tailwind-merge` | ^3.6.0 | ✅ | lib/utils.ts |
| `tailwindcss` | ^4.3.1 | ✅ | Styling |

#### Unused Packages (5)

| Package | Size (approx) | Cleanup Impact |
|---|---|---|
| `@radix-ui/react-dialog` | ~15 KB | Remove — not used; consider if needed for future modals |
| `@radix-ui/react-slot` | ~5 KB | Remove |
| `@radix-ui/react-toast` | ~20 KB | Remove; ErrorAlert serves same purpose |
| `class-variance-authority` | ~10 KB | Remove; Button uses manual variant logic |
| `react-hook-form` | ~50 KB | Remove; form state managed with useState |

#### Dev Dependencies (7 listed)

| Package | Version | Used? |
|---|---|---|
| `@types/node` | ^24.13.2 | ✅ |
| `@types/react` | ^19.2.17 | ✅ |
| `@types/react-dom` | ^19.2.3 | ✅ |
| `@vitejs/plugin-react` | ^6.0.2 | ✅ |
| `oxlint` | ^1.69.0 | ✅ (lint script) |
| `typescript` | ~6.0.2 | ✅ |
| `vite` | ^8.1.0 | ✅ |

## Version Conflict Check

| Layer | Package | Version | Conflict? |
|---|---|---|---|
| Python | torch | >=2.0.0 | None (constrained to major version) |
| Python | transformers | >=4.30.0 | None |
| Node | react | ^19.2.7 | None |
| Node | react-dom | ^19.2.7 | Consistent with react |
| Node | typescript | ~6.0.2 | Latest |

No version conflicts detected.

## Overall Dependency Score: **6.0 / 10**

### Summary
- Python `requirements.txt` is **incomplete** (8 listed, 14+ actually needed)
- Frontend has **5 unused packages** increasing install size and dependency surface
- No lockfile compatibility issues
- No version conflicts

### Priority Actions
1. Add missing Python deps to `requirements.txt`: `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`, `numpy`
2. Remove unused frontend packages: 5 Radix/CVA/RHF packages
3. Pin Python package versions for reproducibility
