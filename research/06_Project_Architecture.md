# 06 — Project Architecture

## 1. Current Pipeline

```mermaid
graph TB
    subgraph "Dataset (CrisisMMD v2.0)"
        TSV[Annotations TSV Files<br/>7 Events]
        IMG[Disaster Images<br/>7 Event Folders]
    end

    subgraph "Data Quality Pipeline"
        DQF[Data Quality Filtering<br/>objective & reproducible rules<br/>no prediction-based removal]
        QCDS[Quality-Controlled Dataset<br/>3,105 samples · duplicates removed]
        SPLIT[Stratified Split 70/15/15<br/>near-duplicate containment · no leakage]
        TRN[TRAIN<br/>2,173 samples]
        VAL[VALIDATION<br/>467 samples]
        TST[TEST<br/>465 samples — evaluation only, excluded from training]
        AUG[Image Augmentation<br/>Resize · HFlip · Rotation · Normalize]
        TOKENIZE[Text Tokenization<br/>DistilBERT Tokenizer<br/>Max 64 tokens]
    end

    subgraph "Image Branch"
        ENET[EfficientNet-B0<br/>Pretrained ImageNet<br/>Frozen Backbone]
        IMG_EMB[Image Embedding<br/>1280-d]
    end

    subgraph "Text Branch"
        DBERT[DistilBERT<br/>Fine-tuned<br/>Uncased]
        TXT_EMB[Text Embedding<br/>768-d]
    end

    subgraph "Fusion Layer"
        CONCAT[Concat Fusion<br/>Concatenate 1280 + 768]
        GATED[Gated Fusion<br/>Learnable Gates + Concatenate]
    end

    subgraph "Classifier"
        FC1[Linear 2048 → 512]
        RELU[ReLU]
        DO[Dropout 0.3]
        FC2[Linear 512 → 3]
        SOFT[Softmax]
    end

    subgraph "Output"
        PRED[Damage Severity<br/>0: Little/No<br/>1: Mild<br/>2: Severe]
    end

    TSV --> DQF
    IMG --> DQF
    DQF --> QCDS
    QCDS --> SPLIT
    SPLIT --> TRN
    SPLIT --> VAL
    SPLIT --> TST
    TRN --> AUG
    VAL --> AUG
    TRN --> TOKENIZE
    VAL --> TOKENIZE
    AUG --> ENET
    ENET --> IMG_EMB
    TOKENIZE --> DBERT
    DBERT --> TXT_EMB
    IMG_EMB --> CONCAT
    TXT_EMB --> CONCAT
    IMG_EMB --> GATED
    TXT_EMB --> GATED
    CONCAT --> FC1
    GATED --> FC1
    FC1 --> RELU
    RELU --> DO
    DO --> FC2
    FC2 --> SOFT
    SOFT --> PRED
```

## 2. Module Architecture

### 2.1 Data Layer

```
backend/datasets/
├── multimodal_data.py     # Load TSVs, filter, clean, deduplicate
├── multimodal_dataset.py  # PyTorch Dataset (image → tensor, text → token IDs)
├── text_preprocessing.py  # URL removal, whitespace normalization
├── text_dataset.py        # PyTorch Dataset for text-only
├── prepare_data.py        # Image dataset loading
├── data_loader.py         # Image data loaders
└── analyze_severity_text.py  # Text analysis utility
```

### 2.2 Model Layer

```
backend/models/
├── fusion_models.py       # ImageFeatureExtractor, TextFeatureExtractor,
                          # ConcatFusion, GatedFusion
├── efficientnet.py        # EfficientNet-B0 builder
└── text_model.py          # DistilBERT builder
```

### 2.3 Training Layer

```
backend/training/
├── fusion_trainer.py      # Fusion training loop
├── fusion_evaluate.py     # Fusion evaluation + metrics
├── fusion_comparison.py   # Model comparison table generator
├── train.py               # Image training loop
├── evaluate.py            # Image evaluation
├── text_trainer.py        # Text training loop
├── text_evaluate.py       # Text evaluation
├── visualize.py           # Plotting (curves, CM, distributions)
└── config.py              # All hyperparameters and paths
```

## 3. Feature Extraction Pipeline

```mermaid
sequenceDiagram
    participant RAW as Raw Data
    participant PRE as Preprocessor
    participant IMG as Image Encoder
    participant TXT as Text Encoder
    participant CACHE as Feature Cache
    participant FUSION as Fusion Trainer

    RAW->>PRE: Load TSV + images + text
    PRE->>PRE: Filter, clean, deduplicate
    PRE->>PRE: Stratified split
    PRE->>IMG: Train/Val/Test images
    PRE->>TXT: Train/Val/Test texts
    IMG->>IMG: EfficientNet-B0 forward
    TXT->>TXT: DistilBERT forward
    IMG->>CACHE: Save 1280-d embeddings
    TXT->>CACHE: Save 768-d embeddings
    CACHE->>FUSION: Load cached features
    FUSION->>FUSION: Train fusion layers only
```

## 4. Training Pipeline

```mermaid
graph LR
    subgraph "Pre-extraction (once)"
        A[Load Images] --> B[EfficientNet-B0]
        B --> C[Cache img_feat.pt]
        D[Load Texts] --> E[DistilBERT]
        E --> F[Cache txt_feat.pt]
    end

    subgraph "Fusion Training (per epoch)"
        G[Load cached features] --> H[Fusion Model]
        H --> I[CrossEntropy Loss]
        I --> J[AdamW Optimizer]
        J --> H
    end

    C --> G
    F --> G
```

## 5. Future Pipeline

```mermaid
graph TB
    subgraph "Future Modules"
        DII[Disaster Impact Index<br/>Weighted severity scoring<br/>0–100 continuous scale]
        XAI[Explainable AI<br/>Grad-CAM for images<br/>Attention for text]
        DI[Decision Intelligence<br/>Risk scoring<br/>Confidence thresholds]
    end

    subgraph "Deployment"
        API[FastAPI Backend<br/>REST endpoints<br/>Model serving]
        UI[React Dashboard<br/>Map visualization<br/>Real-time monitoring]
        DB[(MySQL Database<br/>User management<br/>Prediction history)]
    end

    subgraph "Current Models"
        FUSION[Gated Fusion Model<br/>Best Checkpoint<br/>Macro F1: 0.5375]
    end

    FUSION --> DII
    DII --> XAI
    XAI --> DI
    DI --> API
    API --> UI
    API --> DB
```

## 6. Code Entry Points

| Script | Purpose |
|---|---|
| `run_image_baseline.py` | Train/evaluate image-only EfficientNet-B0 |
| `run_text_baseline.py` | Train/evaluate text-only DistilBERT |
| `run_fusion.py` | Extract features, train/evaluate both fusion models |
| `dataset_audit.py` | Dataset exploration and statistical analysis |
