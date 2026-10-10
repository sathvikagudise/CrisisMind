# 02 — Backend Audit

## Overview

The backend consists of 27 Python files across 6 subdirectories: `api/`, `models/`, `training/`, `datasets/`, `services/` (empty), and `utils/` (empty), plus 4 root-level run scripts.

## 1. API Layer (`backend/api/`)

### Files: 7 (config.py, load_model.py, main.py, predict.py, preprocessing.py, schemas.py, README.md)

### Strengths
- Clean separation of concerns: one file per responsibility
- Pydantic schemas for response validation
- README with curl examples, Docker instructions
- CORS enabled
- File type and size validation on upload
- Startup event loads models once (singleton pattern)

### Issues

| # | Severity | Issue | Location |
|---|---|---|---|
| 1 | **HIGH** | `sys.path.insert(0, ...)` for imports | `main.py:6`, `load_model.py:5` |
| 2 | **HIGH** | Blocking `run_prediction()` in async route | `main.py:82` — event loop blocked under concurrency |
| 3 | **HIGH** | `torch.load` with no error handling | `load_model.py:39` — corrupt checkpoint crashes startup |
| 4 | **MEDIUM** | `response_model=None` disables validation | `main.py:65` |
| 5 | **MEDIUM** | Error responses returned as HTTP 200 with `success=False` | `predict.py:56-60` |
| 6 | **MEDIUM** | Module-level model singletons; no multi-worker support | `main.py:15-17` |
| 7 | **MEDIUM** | `SEVERITY_MAP[idx]` can KeyError | `predict.py:37` |
| 8 | **MEDIUM** | `clean_tweet()` duplicated (3rd copy) | `preprocessing.py:25` |
| 9 | **MEDIUM** | `IMG_TRANSFORM` duplicated | `preprocessing.py:14` vs `multimodal_dataset.py:11` |
| 10 | **MEDIUM** | Config is duplicated with `training/config.py` | `config.py` vs `training/config.py` |
| 11 | **LOW** | Uses `print()` instead of logger | `load_model.py:22-44` (main.py uses logger correctly) |
| 12 | **LOW** | Magic numbers (`1280`, `768`) | `load_model.py:17-18` |

## 2. Models Layer (`backend/models/`)

### Files: 3 (efficientnet.py, fusion_models.py, text_model.py)

### Strengths
- Clean class definitions with proper `nn.Module` inheritance
- ImageFeatureExtractor correctly strips classifier for feature extraction
- GatedFusion implements learnable modality weighting

### Issues

| # | Severity | Issue | Location |
|---|---|---|---|
| 1 | **HIGH** | `ConcatFusion.forward()` returns `logits`; `GatedFusion.forward()` returns `(logits, gates)` tuple — inconsistent API | `fusion_models.py:53,74` |
| 2 | **HIGH** | `count_parameters()` duplicated 3x | `fusion_models.py:95`, `efficientnet.py:23`, `text_model.py:18` |
| 3 | **MEDIUM** | `strict=False` on `load_state_dict` — masks drift | `fusion_models.py:16,36` |
| 4 | **MEDIUM** | Checkpoint loading in `__init__` couples I/O to model definition | `fusion_models.py:12,30` |
| 5 | **MEDIUM** | Cross-package import: `text_model.py` imports from `training/config.py` | `text_model.py:3` |
| 6 | **LOW** | `inplace=True` in Dropout | `efficientnet.py:17` |

## 3. Training Layer (`backend/training/`)

### Files: 8 (config.py, trainer.py, text_trainer.py, fusion_trainer.py, fusion_comparison.py, fusion_evaluate.py, evaluate.py, text_evaluate.py, visualize.py)

### Strengths
- Early stopping implementation
- Class weight computation for imbalanced data
- Metric tracking across epochs
- Comprehensive evaluation functions

### Issues

| # | Severity | Issue | Location |
|---|---|---|---|
| 1 | **CRITICAL** | `EarlyStopping` class duplicated 3x | `trainer.py:8`, `text_trainer.py:10`, `fusion_trainer.py:12` |
| 2 | **CRITICAL** | `compute_class_weights()` duplicated 3x | `trainer.py:33`, `text_trainer.py:34`, `fusion_trainer.py:36` |
| 3 | **HIGH** | Relative paths resolved at CWD import time | `config.py:3-15` — side effects at import |
| 4 | **HIGH** | `save_metrics_csv()` duplicated 2x | `evaluate.py:49`, `text_evaluate.py:50` |
| 5 | **HIGH** | `save_classification_report()` duplicated 2x | `evaluate.py:54`, `text_evaluate.py:55` |
| 6 | **MEDIUM** | Config duplication with `api/config.py` | `config.py` vs `api/config.py` |
| 7 | **MEDIUM** | `mkdir()` side effects at import | `config.py:17-19` |
| 8 | **LOW** | Unused imports (`numpy`, `TEXT_MAX_LEN`) | Multiple files |
| 9 | **LOW** | Inconsistent optimizers (`Adam` vs `AdamW`) | `trainer.py` vs `text_trainer.py`/`fusion_trainer.py` |

## 4. Datasets Layer (`backend/datasets/`)

### Files: 7 (data_loader.py, multimodal_data.py, multimodal_dataset.py, text_dataset.py, text_preprocessing.py, prepare_data.py, analyze_severity_text.py)

### Strengths
- Proper PyTorch Dataset/DataLoader patterns
- Training/validation/test splits from `load_tsv_with_split`
- Image path resolution with root directory joining

### Issues

| # | Severity | Issue | Location |
|---|---|---|---|
| 1 | **HIGH** | `clean_tweet()` duplicated 4x across codebase | `text_preprocessing.py:4`, `multimodal_data.py:9`, `api/preprocessing.py:25` plus run scripts |
| 2 | **MEDIUM** | `load_tsvs()` duplicated | `prepare_data.py:9`, `analyze_severity_text.py:18` |
| 3 | **MEDIUM** | `set_seed()` duplicated | `data_loader.py:9`, `text_dataset.py:8` |
| 4 | **MEDIUM** | `num_workers=0` everywhere — no parallel I/O | All DataLoaders |
| 5 | **MEDIUM** | Blank image on load failure (masks data issues) | `data_loader.py:48`, `multimodal_dataset.py:37` |
| 6 | **MEDIUM** | `pin_memory=False` | All DataLoaders |
| 7 | **MEDIUM** | Missing `__init__.py` | Package not importable |
| 8 | **LOW** | Unused imports (`os`, `numpy`, `Counter`) | Multiple files |

## 5. Root Run Scripts

### Files: 4 (run_fusion.py, run_image_baseline.py, run_text_baseline.py, run_text_eval.py)

### Issues

| # | Severity | Issue | Location |
|---|---|---|---|
| 1 | **CRITICAL** | `train_fusion_cached()` duplicates `fusion_trainer.py:train_fusion()` | `run_fusion.py:104-177` |
| 2 | **HIGH** | `evaluate_fusion_cached()` duplicates `fusion_evaluate.py:evaluate_fusion()` | `run_fusion.py:180-207` |
| 3 | **HIGH** | `load_and_prepare()` duplicated 2x | `run_text_baseline.py:26-50`, `run_text_eval.py:22-44` |
| 4 | **HIGH** | Bug: `plot_split_counts(len(train_df), len(val_df), len(val_df))` **test count = val count** | `run_fusion.py:243` |
| 5 | **MEDIUM** | In-function imports | `run_fusion.py:105-107` |
| 6 | **MEDIUM** | `CachedFusionDataset` and `collate_fusion` defined in script, not in datasets/ | `run_fusion.py:75,92` |

## Summary

### Overall Backend Score: **6.5 / 10**

### Critical Duplication Map
```
clean_tweet() — 4 copies
EarlyStopping — 3 copies
compute_class_weights — 3 copies
count_parameters — 3 copies
save_metrics_csv — 2 copies
save_classification_report — 2 copies
load_tsvs — 2 copies
load_and_prepare — 2 copies
train_fusion — 2 copies (run_fusion.py vs fusion_trainer.py)
IMG_TRANSFORM — 2 copies
Config — 2 copies (api/config.py vs training/config.py)
```

### Key Improvements
1. Extract shared utilities into a `backend/utils/` module (clean_tweet, EarlyStopping, compute_class_weights, count_parameters, save_metrics_csv)
2. Install the project as a package (`pip install -e .`) to eliminate `sys.path.insert`
3. Run blocking inference in a thread pool executor
4. Merge `api/config.py` and `training/config.py` into a single source of truth
5. Remove `run_fusion.py` duplicate code; delegate to `fusion_trainer.py` and `fusion_evaluate.py`
6. Fix `plot_split_counts` bug on line 243 of `run_fusion.py`
