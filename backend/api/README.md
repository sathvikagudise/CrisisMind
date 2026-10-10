# CrisisMind API

Production-ready FastAPI backend serving the Gated Fusion multimodal model for disaster damage severity classification.

## Architecture

```
backend/
├── api/                         # FastAPI application
│   ├── main.py                  # App entry point, routes, middleware
│   ├── predict.py               # Prediction pipeline
│   ├── schemas.py               # Pydantic request/response models
│   ├── load_model.py            # Model loading at startup
│   ├── preprocessing.py         # Image and text preprocessing
│   └── config.py                # API configuration
├── models/                      # Model architectures (reused)
├── training/                    # Training utilities (reused)
├── outputs/checkpoints/         # Trained model weights
│   ├── best_gated_fusion.pth    # Gated Fusion classifier
│   ├── best_model.pth           # EfficientNet-B0 backbone
│   └── best_text_model.pth      # DistilBERT backbone
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Start the server
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at **http://localhost:8000**

## Endpoints

### GET /health

Returns service health status.

```json
{"status": "healthy"}
```

### GET /docs

Swagger UI for interactive API testing.

### POST /predict

Accepts multipart/form-data with `image` (file) and `tweet_text` (string).

#### Example Request (curl)

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@disaster.jpg" \
  -F "tweet_text=Severe damage reported near downtown area"
```

#### Example Response

```json
{
  "success": true,
  "prediction": {
    "severity": "Severe Damage",
    "confidence": 91.84
  },
  "model": "GatedFusion v1",
  "processing_time_ms": 482.0
}
```

#### Error Responses

| HTTP Status | Description |
|---|---|
| 400 | Missing or empty `tweet_text` or `image` |
| 413 | Image exceeds 10 MB |
| 422 | Unsupported image format |
| 500 | Model inference failure |

## Prediction Pipeline

```
Upload Image + Text
       │
       ▼
  ┌─────────────┐
  │  Validate    │  ← Content type, file size, field presence
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │ Preprocess  │  ← Resize 224×224, normalize (image)
  │             │  ← Clean URLs, tokenize max 64 (text)
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │ EfficientNet│  ← Extract 1280-d image embedding
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │ DistilBERT  │  ← Extract 768-d text embedding
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │Gated Fusion │  ← Learnable gates + FC layers
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Softmax    │  ← Severity + confidence
  └──────┬──────┘
         │
         ▼
  Return JSON response
```

## Configuration

All settings in `backend/api/config.py`:

| Setting | Default | Description |
|---|---|---|
| `DEVICE` | auto | `cuda` or `cpu` |
| `MAX_IMAGE_SIZE_MB` | 10 | Maximum upload size |
| `TEXT_MAX_LEN` | 64 | Max BERT tokens |

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r backend/requirements.txt
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t crisismind-api .
docker run -p 8000:8000 crisismind-api
```

## Model Details

- **Model:** Gated Fusion (EfficientNet-B0 + DistilBERT)
- **Inputs:** RGB image (224×224), tweet text (max 64 tokens)
- **Output:** 3-class severity (Little or No Damage, Mild Damage, Severe Damage)
- **Macro F1:** 0.5375
- **Weighted F1:** 0.6415
