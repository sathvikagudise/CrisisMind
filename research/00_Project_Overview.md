# 00 — Project Overview

## 1. Introduction

CrisisMind is a research project focused on automated disaster damage severity assessment using multimodal deep learning. The system leverages paired image and text content from Twitter during natural disaster events to classify infrastructure damage into three severity levels.

## 2. Problem Statement

During natural disasters, social media platforms like Twitter generate massive volumes of multimodal content (images + text) that contain valuable situational awareness information. Manual analysis of this data is time-prohibitive for disaster response teams. Existing automated approaches typically rely on either images or text alone, missing the complementary information available across modalities.

## 3. Research Question

**Can a multimodal fusion framework combining disaster images and tweet text outperform unimodal baselines for automated damage severity classification?**

## 4. Dataset

- **Name:** CrisisMMD v2.0
- **Size:** 18,082 annotated tweet-image pairs
- **Events:** 7 natural disasters (2017)
- **Task:** Damage severity classification (3 classes)
- **Classes:** little_or_no_damage, mild_damage, severe_damage

## 5. Methodology

| Phase | Model | Input | Approach |
|---|---|---|---|
| 1 | EfficientNet-B0 | Image | Transfer learning with frozen backbone |
| 2 | DistilBERT | Text | Fine-tuned transformer classifier |
| 3a | Concat Fusion | Image + Text | Feature concatenation + FC layers |
| 3b | Gated Fusion | Image + Text | Learnable modality gating + FC layers |

## 6. Key Results

- **Gated Fusion** achieves the best performance with Macro F1 of **0.5375**
- Both fusion models outperform unimodal baselines by **+10–27%** in Macro F1
- Learnable gating improves minority class recall over simple concatenation

## 7. Publication Target

- IEEE Conference (ISCRAM, ICWSM, or equivalent)
- Final Year Project Report
