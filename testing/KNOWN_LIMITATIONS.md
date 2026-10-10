# CrisisMind — Known Limitations

## 1. Model / Inference

### 1.1 Text Model Bias
- DistilBERT was fine-tuned on CrisisMMD which is primarily English-language disaster tweets. The model may perform poorly on non-English text, informal slang, or region-specific disaster terminology.
- The text-only analysis tends to over-predict "severe" for emotionally charged language (e.g. "terrible", "devastating") even when the actual damage is minor.

### 1.2 Image Model Bias
- EfficientNet-B0 was trained on CrisisMMD images, which are over-represented by hurricanes (Harvey, Irma, Maria) and under-represented by earthquakes, floods, and tornadoes.
- The model may confuse "mild" and "little" damage categories because the boundary is visually subtle (e.g. "few cracks" vs "moderate cracks").

### 1.3 Confidence Calibration
- The confidence score reflects the model's softmax probability, which is not perfectly calibrated. A confidence of 90% does not guarantee 90% accuracy.
- Confidence tends to be higher for severe damage predictions than for little/mild.

### 1.4 Multimodal Fusion
- When both image and text are provided, the Gated Fusion mechanism averages both modalities. If one modality is noisy (e.g. irrelevant text), the result may be worse than the better single modality.
- No modality weighting is exposed to the user.

---

## 2. Performance

### 2.1 CPU Latency
- All inference runs on CPU by default. On a typical laptop, a single image analysis takes 2–5 seconds. Batch processing of 20 images may take 40–100 seconds.
- For production use, a GPU (CUDA) would reduce latency by 10–20×.

### 2.2 Batch Processing
- The batch queue is in-memory — restarting the server loses all queued jobs.
- No parallel processing — batch items are processed sequentially on the main thread.

### 2.3 Grad-CAM Delay
- Grad-CAM generation adds 1–3 seconds on CPU after the main analysis completes.
- The heatmap overlay is generated at fixed resolution and may not perfectly align with very large or small images.

---

## 3. Data

### 3.1 Dataset Imbalance
- CrisisMMD contains far more "severe_damage" samples (2212) than "little_or_no_damage" (475) or "mild_damage" (839). The model may therefore be biased toward predicting severe damage.
- Some disaster types (e.g. Sri Lanka floods, Mexico earthquake) have very few labeled images, reducing prediction reliability for those events.

### 3.2 No External Database
- All stores (DashboardStore, AnalysisStore, BatchStore) are in-memory. Restarting the server clears all data.
- No persistence, no user accounts, no multi-tenancy.

### 3.3 NASA / External Provider Limitations
- The project references NASA satellite data integration, but this is currently stubbed/mock — no real NASA API calls are made.
- Any external data provider integration would require API keys and rate-limit handling not currently implemented.

---

## 4. Frontend

### 4.1 Mobile Responsiveness
- While the UI is partially responsive, some tables (especially Analysis Center data table) may overflow on small screens (<768px).
- The dashboard charts may be cramped on mobile.

### 4.2 Image Size Limit
- Uploaded images are limited to 10 MB. Larger images are rejected at the API level with a 413 error.

### 4.3 No Offline Support
- The frontend requires a live backend connection. No caching or offline fallback is implemented.

### 4.4 CSV Export Encoding
- Downloaded CSV files use UTF-8 encoding. Some spreadsheet applications (e.g. older Excel) may not display special characters correctly.

---

## 5. Security

### 5.1 No Authentication
- The API has no authentication layer. Anyone who can reach the server port can access all endpoints and data.
- No HTTPS — traffic is plain HTTP.

### 5.2 File Upload Validation
- Image uploads are validated by content-type and file extension only. Malformed files may cause server-side errors.
- Batch CSV files are parsed without strict schema validation beyond column headers.

---

## 6. Deployment

### 6.1 Single Process
- The FastAPI server runs as a single process with uvicorn. No horizontal scaling, no load balancing.
- WebSocket support for real-time batch progress is not implemented (uses polling).

### 6.2 Environment
- Python 3.11+ required. Some dependencies (torch, torchvision, transformers) are large (~2 GB total).
- Frontend build requires Node.js 18+.
