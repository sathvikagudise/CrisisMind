# 09 — Security Audit

## 1. Input Validation

| Endpoint | Validation | Issues |
|---|---|---|
| `POST /predict` — image | ✅ File type whitelist (JPEG, PNG, WebP, BMP) | Line 87 in main.py checks `image.content_type` against `ALLOWED_IMAGE_TYPES`. **Note**: content_type is user-controlled and can be spoofed. |
| `POST /predict` — image | ✅ File size limit (10 MB) | Line 91 checks `len(await image.read())`. Reads entire file into memory first. |
| `POST /predict` — text | ✅ Not empty | Lines 82-83 check `not tweet_text` |
| `POST /predict` — text | ❌ No sanitization | User text passed directly to DistilBERT tokenizer. Risk of adversarial inputs but not injection (no SQL/exec). |
| `POST /predict` — text | ✅ Length limited | FastAPI's `Form()` with no explicit max but truncated in preprocessing |
| `GET /health` | N/A | No parameters |

## 2. File Upload Safety

| Concern | Status | Notes |
|---|---|---|
| MIME type spoofing | ⚠️ **Partial** | `content_type` is HTTP header, easily faked. Should validate file magic bytes (PIL will catch this on load). |
| Path traversal in filename | ✅ **Safe** | Filename not used for server-side storage; only passed to PIL for in-memory processing |
| Zip bombs / decompression bombs | ⚠️ **Partial** | PIL can be vulnerable to decompression bombs. The 10 MB limit provides partial protection. |
| Image parsing crash | ⚠️ **Partial** | No try/except around PIL `Image.open()`. Malformed image crashes the request (500). |
| File type validation via content | ✅ **Good** | PIL's `Image.open()` will reject non-image files, providing second-layer validation |

## 3. Path Traversal

| Risk | Status |
|---|---|
| Checkpoint filenames from user input | ❌ **N/A** — filenames are hardcoded in config |
| TSV file paths from user input | ❌ **N/A** — paths are hardcoded |
| Image path resolution with `..` | ✅ **Safe** — image paths are constructed from hardcoded dataset dir + TSV entry; no user-controlled path components |

## 4. CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ⚠️ Wide open
    allow_credentials=True,
    allow_methods=["*"],       # ⚠️ All methods
    allow_headers=["*"],       # ⚠️ All headers
)
```

| Issue | Severity | Notes |
|---|---|---|
| `allow_origins=["*"]` | **MEDIUM** | Acceptable for demo/research; must be restricted in production |
| `allow_credentials=True` with `*` | **LOW** | Invalid per spec (credentials+wildcard don't work in browsers) |

## 5. Exception Handling

| Error Source | Caught? | Response |
|---|---|---|
| Missing image field | ✅ FastAPI validation | 422 |
| Missing text field | ✅ FastAPI validation | 422 |
| Invalid image type (FastAPI) | ✅ Custom check | 422 |
| File too large | ✅ Custom check | 413 |
| Prediction failure (model error) | ✅ try/except | 500 |
| Load model failure (startup) | ❌ **Not caught** | Raw error exposed, server exits |
| PIL image load error | ❌ **Not caught** | Unhandled → 500 with traceback |
| Corrupt checkpoint | ❌ **Not caught** | Server crashes on startup |

## 6. Information Exposure

| Concern | Status |
|---|---|
| Stack traces in production | ⚠️ **Partial** — FastAPI debug mode disabled by default |
| Model architecture exposed | ✅ Only through `/predict` response (model name string) |
| Dataset paths exposed | ✅ Not returned in any response |
| Internal directory structure | ✅ Not exposed |

## 7. Additional Recommendations

| Priority | Recommendation |
|---|---|
| HIGH | Add try/except around PIL `Image.open()` in preprocessing |
| HIGH | Validate file content (magic bytes) before PIL processing |
| MEDIUM | Restrict CORS origins for production |
| MEDIUM | Implement startup error handling (graceful shutdown on model load failure) |
| LOW | Add request rate limiting |
| LOW | Add request size middleware (not just per-file) |

## Overall Security Score: **7.0 / 10**
