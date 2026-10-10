# 05 — Experimental Results

## 1. Comprehensive Comparison Table

| Model | Accuracy | Precision (Macro) | Recall (Macro) | **F1 (Macro)** | Precision (Weighted) | Recall (Weighted) | **F1 (Weighted)** |
|---|---|---|---|---|---|---|---|
| Image Baseline | 0.5841 | 0.4809 | 0.5063 | **0.4869** | 0.6154 | 0.5841 | **0.5959** |
| Text Baseline | 0.5250 | 0.4291 | 0.4817 | **0.4221** | 0.5741 | 0.5250 | **0.5328** |
| Concat Fusion | 0.6371 | 0.5332 | 0.5287 | **0.5302** | 0.6439 | 0.6371 | **0.6400** |
| **Gated Fusion** 🏆 | 0.6352 | **0.5329** | **0.5437** | **0.5375** | **0.6493** | 0.6352 | **0.6415** |

🏆 **Best overall: Gated Fusion** (highest Macro F1 = 0.5375)

## 2. Class-Wise F1 Comparison

| Class | Image | Text | Concat | Gated | Best |
|---|---|---|---|---|---|
| little_or_no_damage | 0.3778 | 0.3721 | 0.4118 | **0.4161** | Gated 🏆 |
| mild_damage | 0.3454 | 0.2135 | 0.3985 | **0.4242** | Gated 🏆 |
| severe_damage | 0.7377 | 0.6808 | **0.7805** | 0.7721 | Concat 🏆 |

## 3. Relative Improvement Over Baselines

### Gated Fusion vs Image Baseline

| Metric | Image | Gated | Δ |
|---|---|---|---|
| Accuracy | 0.5841 | 0.6352 | **+8.7%** |
| F1 Macro | 0.4869 | 0.5375 | **+10.4%** |
| F1 Weighted | 0.5959 | 0.6415 | **+7.7%** |
| F1 (little_or_no_damage) | 0.3778 | 0.4161 | **+10.1%** |
| F1 (mild_damage) | 0.3454 | 0.4242 | **+22.8%** |
| F1 (severe_damage) | 0.7377 | 0.7721 | **+4.7%** |

### Gated Fusion vs Text Baseline

| Metric | Text | Gated | Δ |
|---|---|---|---|
| Accuracy | 0.5250 | 0.6352 | **+21.0%** |
| F1 Macro | 0.4221 | 0.5375 | **+27.3%** |
| F1 Weighted | 0.5328 | 0.6415 | **+20.4%** |
| F1 (little_or_no_damage) | 0.3721 | 0.4161 | **+11.8%** |
| F1 (mild_damage) | 0.2135 | 0.4242 | **+98.7%** |
| F1 (severe_damage) | 0.6808 | 0.7721 | **+13.4%** |

## 4. Precision-Recall Across Classes

| Model | little_or_no_damage | mild_damage | severe_damage |
|---|---|---|---|
| | P | R | F1 | P | R | F1 | P | R | F1 |
| Image Baseline | 0.3119 | 0.4789 | 0.3778 | 0.3496 | 0.3413 | 0.3454 | 0.7811 | 0.6988 | 0.7377 |
| Text Baseline | 0.2614 | 0.6452 | 0.3721 | 0.2836 | 0.1712 | 0.2135 | 0.7423 | 0.6287 | 0.6808 |
| Concat Fusion | 0.4308 | 0.3944 | 0.4118 | 0.3786 | 0.4206 | 0.3985 | 0.7901 | 0.7711 | 0.7805 |
| **Gated Fusion** | 0.3974 | 0.4366 | **0.4161** | 0.4058 | 0.4444 | **0.4242** | 0.7955 | 0.7500 | **0.7721** |

## 5. Visual Comparison

### Confusion Matrices

| Model | Confusion Matrix |
|---|---|
| Image Baseline | ![Image CM](assets/confusion_matrices/image_baseline_cm.png) |
| Text Baseline | ![Text CM](assets/confusion_matrices/text_baseline_cm.png) |
| Concat Fusion | ![Concat CM](assets/confusion_matrices/concat_fusion_cm.png) |
| Gated Fusion | ![Gated CM](assets/confusion_matrices/gated_fusion_cm.png) |

## 6. Discussion

### Key Findings

1. **Multimodal fusion consistently outperforms unimodal approaches** — Both fusion models achieve higher Macro F1 than either baseline, confirming the hypothesis that combining image and text modalities provides complementary information for damage assessment.

2. **Gated Fusion achieves the best overall Macro F1 (0.5375)** — The gating mechanism's ability to adaptively weight modalities per-sample leads to improved performance, particularly for minority classes.

3. **Minority class F1 gains are substantial** — mild_damage F1 improves by **22.8%** over the image baseline and **98.7%** over the text baseline, suggesting that multimodal fusion provides the most benefit for ambiguous or underrepresented cases.

4. **Text baseline underperforms image baseline** — DistilBERT's lower Macro F1 (0.4221 vs 0.4869) suggests that tweet text alone is less reliable for severity assessment than visual content.

5. **severe_damage is consistently well-predicted** — All models achieve F1 > 0.68 for this class, likely because severe structural damage produces both distinctive visual patterns and strong textual descriptors.

### Why Gated Fusion Wins

The Gated Fusion model's advantage comes from:
- **Per-sample modality weighting** instead of fixed concatenation
- **Improved minority class recall** through adaptive information selection
- **Noise suppression** from the sigmoid gating mechanism
- **Better handling of modality mismatch** (e.g., generic tweet + specific image or vice versa)

### Limitations

1. All models struggle with the **mild_damage** class (best F1 = 0.4242), suggesting the boundary between mild and other classes is inherently ambiguous
2. The **little_or_no_damage** class has limited support (71 test samples), making metrics less reliable
3. Training is limited by overall dataset size (3,526 severity samples)
4. Feature extractors are frozen — end-to-end fine-tuning could improve performance with sufficient data
