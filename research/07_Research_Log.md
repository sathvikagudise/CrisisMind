# 07 — Research Log

## 2026-06-23 — Dataset Analysis (Phase 0)

**Completed:**
- Explored CrisisMMD v2.0 dataset structure
- Loaded and analyzed all 7 TSV annotation files
- Verified all 18,082 image paths exist
- Generated class distribution, humanitarian category, and event-wise charts
- Created `dataset_audit.py` and `dataset_summary_report.md`

**Key Findings:**
- 18,082 total samples, 7 disaster events
- 3,627 severity-annotated samples (after filtering)
- Significant class imbalance: severe_damage (62.74%) vs little_or_no_damage (13.47%)
- Zero missing images — all paths valid

---

## 2026-06-23 — Image Baseline Model (Phase 1)

**Completed:**
- Built EfficientNet-B0 with frozen ImageNet backbone
- Created modular code structure under `backend/`
- Trained for 5 epochs (CPU — hardware limited)
- Achieved: Accuracy 0.5425, Macro F1 0.4417

**Problems Faced:**
- CPU training extremely slow (killed by timeout multiple times)
- DataLoader hanging on Windows — resolved by using `num_workers=0`
- Class imbalance causing poor minority class performance

**Solutions:**
- Reduced batch size and image resolution for demo run
- Verified pipeline works end-to-end
- Restored config to production-ready defaults

**Next Steps:**
- Build text baseline model
- Implement multimodal fusion

---

## 2026-06-25 — Text Baseline Model (Phase 2)

**Completed:**
- Built DistilBERT text classifier
- Text preprocessing: URL removal, whitespace normalization, hashtag preservation
- Tokenization with DistilBertTokenizer (max 64 tokens)
- Full fine-tuning of all DistilBERT parameters
- Achieved: Accuracy 0.5250, Macro F1 0.4221

**Problems Faced:**
- Transformer model training killed by OOM on CPU
- Full training run interrupted — checkpoint saved but metrics not generated

**Solutions:**
- Created `run_text_eval.py` to load saved checkpoint and run evaluation separately
- DistilBERT checkpoint (267MB) preserved successfully
- Text deduplication identified 328 duplicate tweets

**Observations:**
- Text baseline underperforms image baseline (Macro F1: 0.4221 vs 0.4869)
- mild_damage is hardest class for text (F1 = 0.2135)
- High recall but low precision for little_or_no_damage

---

## 2026-06-26 — Multimodal Fusion (Phase 3)

**Completed:**
- Pre-extracted image features (1280-d) using trained EfficientNet-B0
- Pre-extracted text features (768-d) using trained DistilBERT
- Built two fusion models:
  - **Concat Fusion**: Simple concatenation → FC layers
  - **Gated Fusion**: Learnable modality gating → FC layers
- Feature caching strategy dramatically reduced training time
- Full evaluation and comparison report generated

**Results:**
| Model | Macro F1 | Weighted F1 |
|---|---|---|
| Image Baseline | 0.4869 | 0.5959 |
| Text Baseline | 0.4221 | 0.5328 |
| Concat Fusion | 0.5302 | 0.6400 |
| **Gated Fusion 🏆** | **0.5375** | **0.6415** |

**Problems Faced:**
- Training killed by timeout when running end-to-end (EfficientNet + DistilBERT per batch)
- `isinstance(model, ConcatFusion)` import error in fusion_trainer.py
- Comparison report crashed due to missing `tabulate` package

**Solutions:**
- Implemented **pre-extraction + caching** strategy (features saved to disk)
- Fixed import: added `from backend.models.fusion_models import ConcatFusion`
- Installed `tabulate` package
- Both fusion models trained in under 2 minutes using cached features

**Observations:**
- Both fusion models outperform unimodal baselines
- Gated Fusion achieves best Macro F1 (+10.4% over image, +27.3% over text)
- Mild_damage class shows largest improvement from fusion

---

## 2026-06-26 — Research Documentation

**Completed:**
- Created comprehensive research knowledge base (`research/`)
- Documented all 4 phases with exact experimental results
- Copied all generated assets into organized subdirectories
- Created comparison tables, architecture diagrams, novelty statement
- All 10 documentation tasks completed

**Next Steps (Planned):**
- Phase 4: Disaster Impact Index (DII) — weighted severity scoring
- Phase 5: Explainable AI — Grad-CAM + attention visualization
- Phase 6: Decision Intelligence — risk scoring + confidence thresholds
- Phase 7: Deployment — FastAPI + React + MySQL
