# 🧠 Multimodal Document Q&A System<div align="center">



A powerful AI-powered document analysis system that combines text and image processing with advanced retrieval techniques to answer questions from PDF documents intelligently.# Fusion QA: Early, Late, and Hybrid Retrieval with Gemini + GPU



[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)Multi-modal, GPU-accelerated question answering over PDFs with three fusion strategies, a unified API, and a Streamlit chat UI.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)</div>

[![CUDA](https://img.shields.io/badge/CUDA-Supported-brightgreen.svg)](https://developer.nvidia.com/cuda-zone)

## Highlights

## ✨ Key Features

- Multi-modal retrieval: text (E5) + images (CLIP) with optional OCR

- **📄 Intelligent PDF Processing**: Extracts text, images, tables, and charts with structure preservation- Three fusion modes: Late, Early, Hybrid (unified dispatcher on one endpoint)

- **🔍 Three Fusion Strategies**:- GPU acceleration: CUDA-backed embeddings and reranking, optional FAISS GPU

  - **Late Fusion**: Separate text and image retrieval, merged at answer generation- Answer generation with Gemini; robust retry/backoff and fallback model

  - **Early Fusion**: Combined text+image embeddings for unified semantic search- Streamlit chat UI with URL/upload, image previews, health badges, and presets

  - **Hybrid Fusion**: Three-way retrieval with adaptive score combination (best performance)- Evaluation harness with EM/Precision/Recall/F1, BLEU-1, ROUGE-L, and numeric-aware metrics

- **🖼️ Multimodal Understanding**: Advanced OCR for tables, CLIP-based image analysis

- **🤖 Smart Responses**: Context-aware answer generation with Google Gemini## Tech stack

  - Detailed summaries (400-600 words)

  - Concise facts (50-150 words)- Backend: FastAPI (Python)

  - Balanced explanations (150-300 words)- Models: Sentence-Transformers (E5 text, CLIP ViT-L-14 text+image), CrossEncoder (MS MARCO)

- **⚡ GPU Acceleration**: CUDA support for embeddings, FAISS indexing, and model inference- Vector search: FAISS (L2-normalized cosine/IP; optional GPU)

- **🎨 Interactive Web UI**: Modern Streamlit dashboard with real-time feedback- LLM: Google Gemini (default `gemini-2.0-flash-exp`; configurable)

- **⚙️ Highly Configurable**: API URL, fusion modes, image inclusion via UI settings- PDF/image: PyMuPDF (fitz), PIL; optional Tesseract OCR

- **🛡️ Production-Ready**: Rate limiting, caching, circuit breaker, error handling- UI: Streamlit

- Infra: Windows + PowerShell scripts, .env config

## 🏗️ System Architecture

## Architecture (end-to-end)

```

┌─────────────────────────────────────────────────────────────┐```mermaid

│                    Streamlit Frontend                        │flowchart LR

│                   (app.py - Port 8501)                       │    subgraph Client

│  • PDF Upload/URL Input  • Settings Panel  • Results View   │      UI[Streamlit Chat UI]

└───────────────────────────┬─────────────────────────────────┘    end

                            │ HTTP REST API    subgraph API

┌───────────────────────────▼─────────────────────────────────┐      U[Unified API /hackrx/run]

│                    Unified API Gateway                       │      LF[(Late Fusion)]

│                 (main_api.py - Port 8000)                    │      EF[(Early Fusion)]

│            Routes requests to fusion strategies              │      HF[(Hybrid Fusion)]

└────────┬──────────────┬──────────────┬──────────────────────┘    end

         │              │              │    subgraph ModelServer

         ▼              ▼              ▼      T[Text Encoder E5]

┌─────────────┐  ┌─────────────┐  ┌─────────────┐      C[CLIP Encoder]

│Late Fusion  │  │Early Fusion │  │Hybrid Fusion│      R[Cross-Encoder]

│             │  │             │  │             │    end

│Text + Image │  │Fused Embed  │  │Three-way    │    PDF[(PDF URL or Upload)]

│Separate     │  │Single Index │  │Combination  │    UI -- asks --> U

└──────┬──────┘  └──────┬──────┘  └──────┬──────┘    U -- fetch/parse --> PDF

       │                │                │    U -- embeddings --> T

       └────────────────┴────────────────┘    U -- embeddings --> C

                        │    U -- rerank --> R

         ┌──────────────▼───────────────┐    U -- answer --> G[(Gemini)]

         │      Model Server             │    U -- result --> UI

         │  (model_server.py - 8001)    │    U <-. dispatch .-> LF

         │                               │    U <-. dispatch .-> EF

         │  • E5 Text Embeddings        │    U <-. dispatch .-> HF

         │  • CLIP Image Embeddings     │```

         │  • CrossEncoder Reranking    │

         │  • GPU/CPU Auto-detection    │## Fusion modes explained

         └───────────────────────────────┘

```### Late Fusion



## 🚀 Quick StartSeparate retrieval pipelines per modality, then merge and rerank.



### Prerequisites```mermaid

flowchart TB

- **Python 3.9+** (3.10 or 3.11 recommended)    Q[Query] --> TE[E5 text emb]

- **CUDA-capable GPU** (optional, but recommended for speed)    Q --> CT[CLIP text emb]

- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))    D[Per-page text] --> TFAISS[FAISS text index]

    I[Page images] --> CE[CLIP image emb]

### Installation    CE --> IFAISS[FAISS image index]

    TE --> TFAISS

1. **Clone the repository**    CT --> IFAISS

```bash    TFAISS --> TK1[Top-K text]

git clone https://github.com/yourusername/multimodal-qa-system.git    IFAISS --> TK2[Top-K images]

<div align="center">

# 🧠 Multimodal Document Q&A System

AI-powered question answering over PDFs that understands both text and images. Includes three fusion strategies (Late, Early, Hybrid), a unified API, a GPU-ready model server, and a clean Streamlit UI.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-brightgreen.svg)](https://developer.nvidia.com/cuda-zone)

</div>

## ✨ What’s inside

- Multimodal retrieval: text (E5) + images (CLIP), with optional OCR for tables/figures
- Three fusion modes: Late, Early, Hybrid — selectable from UI or API
- Unified FastAPI endpoint with detailed debug info per question
- GPU-ready model server for embeddings and reranking (E5, CLIP ViT-L-14, CrossEncoder)
- Streamlit dashboard with settings panel, health checks, previews, and downloads
- Gemini-powered answers with rate limiting, retries, caching, and circuit breaker

## 🧩 How it works

```
┌─────────────────────────────────────────┐       ┌───────────────────────────┐
│        Streamlit Frontend (8501)        │       │   Model Server (8001)     │
│  • Upload/URL • Fusion mode • Settings  │       │ • E5 text embeddings      │
│  • Ask Qs • Results • Debug • Preview   │       │ • CLIP image embeddings    │
└───────────────┬─────────────────────────┘       │ • CrossEncoder reranking   │
                │  HTTP                           └──────────────┬────────────┘
┌───────────────▼─────────────────────────┐                     HTTP
│          Unified API (8000)             │◀───────────────────────────────┐
│  /hackrx/run  → late | early | hybrid   │                                │
│  /hackrx/health, /hackrx/fusions        │                                │
└─────────────────────────────────────────┘                                │
                │                                                          │
                └── Builds contexts (text + images) → Gemini → Answers ────┘
```

## 🚀 Quick start

### 1) Install

- Python 3.9+ (3.10/3.11 recommended)
- Optional: NVIDIA GPU + CUDA for best performance

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your .env from template and add your Gemini key:

```powershell
copy .env.example .env    # Windows
# or
cp .env.example .env      # Linux/Mac
# then edit .env and set GEMINI_API_KEY
```

### 2) Start everything (Windows)

```powershell
./start_unified.ps1
```

This frees ports 8000/8001/8501, then starts:
- Model Server (http://localhost:8001/health)
- Unified API (http://localhost:8000/hackrx/health)
- Streamlit UI (http://localhost:8501)

### 2b) Manual start (Linux/Mac or custom)

```bash
# Terminal 1
uvicorn model_server:app --host 127.0.0.1 --port 8001

# Terminal 2
uvicorn main_api:app --host 127.0.0.1 --port 8000

# Terminal 3
streamlit run app.py --server.port 8501
```

## 🖥️ Using the UI

1. Open http://localhost:8501
2. In the sidebar:
   - Pick “URL” or “Upload PDF”
   - Choose fusion mode: late | early | hybrid
   - Optional: toggle “Include images in answers”
   - Settings → set API Base URL if your API runs elsewhere
3. Ask a question and click “Run Q&A”
4. Explore tabs: Answers, Debug (hits/scores), and Preview (images used)

## 🔗 Unified API

- List fusion modes: GET `http://localhost:8000/hackrx/fusions`
- Health: GET `http://localhost:8000/hackrx/health`
- Run Q&A: POST `http://localhost:8000/hackrx/run`

Request body:

```json
{
  "documents": "<PDF path or URL>",
  "questions": ["Summarize the report"],
  "fusion": "hybrid",  // late | early | hybrid
  "include_images": true
}
```

Response shape:

```json
{
  "answers": ["..."],
  "debug": { "per_q": [...], "pages": 42, "images": 10, "fusion": "hybrid" }
}
```

## 🧠 Model server endpoints (8001)

- GET `/health` → { status, device, device_id }
- POST `/get_text_embeddings` → { embeddings }
- POST `/get_clip_text_embeddings` → { embeddings }
- POST `/get_image_embeddings` → { embeddings }
- POST `/rerank` → { scores }

## 🤖 Answer style: automatic and concise

The system adapts answer length to the question:

- Summary queries → 400–600 words, comprehensive with data cites
- Fact/value queries → 50–150 words, direct answer first with brief context
- Analytical questions → 150–300 words, balanced with numbers and citations

## ⚙️ Configuration (.env)

Essential keys:

- GEMINI_API_KEY=your_api_key
- MODEL_SERVER_URL=http://127.0.0.1:8001
- UNIFIED_URL=http://127.0.0.1:8000

Models and devices:

- MODEL_DEVICE=cuda | cpu
- MODEL_DEVICE_ID=0
- TEXT_EMB_MODEL_NAME=intfloat/e5-base
- CLIP_MODEL_NAME=clip-ViT-L-14
- CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
- FAISS_USE_GPU=true | false

Gemini behavior and stability:

- GEMINI_MODEL=gemini-2.0-flash-exp
- GEMINI_FALLBACK_MODEL=gemini-1.5-flash (optional)
- GEMINI_INCLUDE_IMAGES=true | false
- GEMINI_MAX_RETRIES=3
- GEMINI_BACKOFF_BASE=5.0
- GEMINI_BACKOFF_MAX=60.0
- GEMINI_CONCURRENCY=2
- GEMINI_CIRCUIT_FAILS=6
- GEMINI_CIRCUIT_RESET=120

Retrieval knobs (set via env for each fusion module as needed):

- K_TEXT, K_IMAGE, K_FUSED
- MAX_TEXT_CHARS
- FUSION_TEXT_WEIGHT, FUSION_IMAGE_WEIGHT
- HYBRID_W_TEXT, HYBRID_W_IMAGE, HYBRID_W_FUSED

## 📦 Project structure

```
app.py                      # Streamlit UI with settings panel
main_api.py                 # Unified FastAPI (dispatch to late/early/hybrid)
main_latefusion.py          # Late fusion (separate text/image retrieval, rerank)
main_earlyfusion.py         # Early fusion (fused per-page vectors)
main_hybridfusion.py        # Hybrid fusion (blend text/image/fused scores)
model_server.py             # Embeddings + rerank service (E5, CLIP, CrossEncoder)
utils.py                    # Model server clients + Gemini with caching/backoff
start_unified.ps1           # One-click startup (Windows)
requirements.txt            # Pinned dependencies
verify_setup.py             # Sanity check script
_lf_cache/                  # Run cache for extracted images (auto-created)
.gemini_cache/              # Gemini response cache (auto-created)
```

## 🧪 Verify setup

```powershell
python verify_setup.py
```

Checks your .env, packages, CUDA/FAISS, and key project files.

## 🐛 Troubleshooting

- Ports busy? Use `stop_all.ps1` (Windows) or kill processes on 8000/8001/8501
- CUDA OOM or no GPU? Set `MODEL_DEVICE=cpu` and `FAISS_USE_GPU=false`
- Gemini 429/quota? Retries/backoff included; set `GEMINI_FALLBACK_MODEL`
- Slow? Reduce K values; disable OCR if not needed

## 📝 License

MIT License — see `LICENSE`.

## 🙌 Acknowledgments

- Google Gemini • Sentence-Transformers • FAISS • FastAPI • Streamlit • PyMuPDF

---

If this project helps you, a star on GitHub means a lot.
Model server and retrieval:
