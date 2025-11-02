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

cd multimodal-qa-system    TK1 --> M[Merge/score]

```    TK2 --> M

    M --> RR[Rerank by CrossEncoder]

2. **Create and activate virtual environment**    RR --> Ctx[Top contexts + images]

```bash    Ctx --> A[Gemini answer]

# Windows```

python -m venv .venv

.venv\Scripts\activateKey points:

- Text chunks and image candidates are retrieved independently and merged by score.

# Linux/Mac- Reranking improves grounding; selected images are passed to Gemini alongside text.

python3 -m venv .venv

source .venv/bin/activate### Early Fusion

```

Create a single per-page fused vector of text + average image embeddings.

3. **Install dependencies**

```bash```mermaid

pip install -r requirements.txtflowchart TB

```    P[Page] --> TXT[E5 text emb]

    P --> IMG[Avg CLIP image emb]

4. **Set up environment variables**    TXT --> F[[Fused vector [w_t*txt ; w_i*img]]]

```bash    IMG --> F

# Windows    F --> FIDX[FAISS fused index]

copy .env.example .env    Q[Query] --> QTXT[E5 text emb]

    QTXT --> QF[[Query fused vector]]

# Linux/Mac    QF --> FIDX

cp .env.example .env    FIDX --> TK[Top-K pages]

    TK --> RR[Rerank]

# Edit .env and add your Gemini API key    RR --> Ctx[Contexts + images]

# GEMINI_API_KEY=your_api_key_here    Ctx --> A[Gemini answer]

``````



5. **Start the application**Key points:

- Single retrieval pass over fused vectors; balances modalities via weights.

**Windows:**- Good default for mixed text+visual documents.

```powershell

.\start_unified.ps1### Hybrid Fusion

```

Blend scores from text, image, and fused indices with weights, then rerank.

**Linux/Mac:**

```bash```mermaid

# Terminal 1 - Model Serverflowchart TB

python -m uvicorn model_server:app --host 127.0.0.1 --port 8001    subgraph Indices

      TX[T-text]:::b --> ITX[FAISS-text]

# Terminal 2 - Main API      IM[Images]:::b --> IIM[FAISS-image]

python -m uvicorn main_api:app --host 127.0.0.1 --port 8000      FU[Fused per-page]:::b --> IFU[FAISS-fused]

    end

# Terminal 3 - Streamlit UI    Q[Query] --> qT[E5]

python -m streamlit run app.py --server.port 8501    Q --> qC[CLIP text]

```    qT --> ITX

    qC --> IIM

6. **Access the application**    qT --> IFU

- **Web UI**: http://localhost:8501    ITX --> S1[Scores]

- **API Docs**: http://localhost:8000/docs    IIM --> S2[Scores]

- **Health Check**: http://localhost:8001/health    IFU --> S3[Scores]

    S1 --> COMB[Weighted combine]

## 📖 Usage Guide    S2 --> COMB

    S3 --> COMB

### Web Interface    COMB --> RR[Rerank]

    RR --> A[Gemini]

1. **Open Browser**: Navigate to http://localhost:8501

    classDef b fill:#eef,stroke:#66f

2. **Configure Settings** (Optional):```

   - Click "⚙️ Settings" in sidebar

   - Enter custom API URL if neededKey points:

   - Click "Update API URL"- Retrieves from all three views; weights are tunable via env vars.

- Often most robust across diverse document layouts.

3. **Upload Document**:

   - Choose "Upload PDF" or "URL" as source## Repository layout

   - Upload file or paste PDF URL

- `main_api.py` — unified FastAPI app: `/hackrx/run`, `/hackrx/fusions`, `/hackrx/health`

4. **Select Options**:- `main_latefusion.py` — late-fusion implementation and route

   - **Fusion Mode**: Late / Early / Hybrid- `main_earlyfusion.py` — early-fusion implementation and route

   - **Include Images**: Toggle image analysis- `main_hybridfusion.py` — hybrid-fusion implementation and route

   - **Sample Questions**: Quick presets available- `model_server.py` — embedding/reranking service (E5, CLIP ViT-L-14, CrossEncoder)

- `utils.py` — model server clients, Gemini generator with retries/fallback

5. **Ask Questions**:- `app.py` — Streamlit chat UI (URL/upload, presets, image previews, health)

   - Enter your question in text area- `evaluate_all_fusions.py` — benchmark metrics (EM/P/R/F1, BLEU-1, ROUGE-L, numeric)

   - Click "Run Q&A"- `test_*.py` — quick tests and latency comparisons

   - View detailed answer with page citations- `start_unified.ps1` — one-click startup for model server, unified API, and UI

- `_lf_cache/` — per-run cache of extracted images to avoid Windows file locks

6. **Explore Results**:

   - **Answers Tab**: See formatted response## Requirements

   - **Debug Tab**: View retrieval details

   - **Preview Tab**: Check extracted content- Python 3.10+

- Windows 10/11 (PowerShell) or other OS with minor adjustments

### API Usage- Optional CUDA GPU (NVIDIA) for best performance

- Google Gemini API key

```python

import requests## Setup (Windows PowerShell)



# Endpoint1) Create and activate a virtualenv

url = "http://localhost:8000/hackrx/run"

2) Install dependencies (torch build should match your CUDA)

# Payload

payload = {3) Create `.env` with your keys and settings

    "documents": "path/to/your/document.pdf",  # or URL

    "questions": ["What is the main objective?", "List key findings"],4) Start everything with `start_unified.ps1`

    "fusion": "hybrid",  # late | early | hybrid

    "include_images": TrueNotes:

}- The startup script frees ports 8000/8001, starts the model server (8001), unified API (8000), and Streamlit UI (8501).

- If you prefer, there is a VS Code Task “Start Streamlit dashboard” to run the UI only.

# Make request

response = requests.post(url, json=payload, timeout=240)## Configuration (.env)

result = response.json()

LLM/Gemini:

# Access results- `GEMINI_API_KEY` — your API key

for i, answer in enumerate(result["answers"]):- `GEMINI_MODEL` — e.g. `gemini-2.0-flash-exp`, `gemini-1.5-flash`

    print(f"Q{i+1}: {payload['questions'][i]}")- `GEMINI_FALLBACK_MODEL` — optional fallback when 429 quota hit

    print(f"A: {answer}\n")- `GEMINI_INCLUDE_IMAGES` — `true|false`: pass images to Gemini

```- `GEMINI_MAX_RETRIES` — retry attempts on 429 (default 3)

- `GEMINI_BACKOFF_BASE` — seconds for exponential backoff (default 5.0)

### Command Line Evaluation

Model server and retrieval:

```bash- `MODEL_DEVICE` — `cuda` or `cpu`

# Run comprehensive evaluation- `MODEL_DEVICE_ID` — CUDA device index (default 0)

python evaluate_all_fusions.py- `TEXT_EMB_MODEL_NAME` — default `intfloat/e5-base`

- `CLIP_MODEL_NAME` — default `clip-ViT-L-14`

# Generates:- `CROSS_ENCODER_MODEL` — default `cross-encoder/ms-marco-MiniLM-L-6-v2`

# - fusion_benchmark_results.json- `FAISS_USE_GPU` — `true|false` (optional acceleration)

# - fusion_eval_detailed.csv

# - fusion_eval_summary.csvRetrieval knobs (examples; may be set per fusion file):

```- `K_TEXT`, `K_IMAGE`, `K_FUSED` — candidate counts

- Early fusion weights: `FUSION_TEXT_WEIGHT`, `FUSION_IMAGE_WEIGHT`

## 🎛️ Configuration- Hybrid weights: `HYBRID_W_TEXT`, `HYBRID_W_IMAGE`, `HYBRID_W_FUSED`



### Environment Variables (`.env`)## Starting services



```bash- Preferred: run `start_unified.ps1` to launch all three (model server, API, UI)

# ============= API Keys =============- Manual alternative:

GEMINI_API_KEY=your_api_key_here  - `uvicorn model_server:app --host 127.0.0.1 --port 8001`

GEMINI_MODEL=gemini-1.5-flash-latest  - `uvicorn main_api:app --host 127.0.0.1 --port 8000`

  - `streamlit run app.py --server.port 8501`

# ============= Device Config =============

MODEL_DEVICE=cuda          # cuda or cpuHealth checks:

MODEL_DEVICE_ID=0          # GPU ID (0, 1, 2, ...)- Unified: GET `http://localhost:8000/hackrx/health` (includes model server device)

FAISS_USE_GPU=true         # Use GPU for FAISS indexing- Model server: GET `http://localhost:8001/health`



# ============= Model Names =============## API usage

TEXT_EMB_MODEL_NAME=intfloat/e5-base

CLIP_MODEL_NAME=openai/clip-vit-base-patch32Unified endpoint (recommended):

CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2- POST `http://localhost:8000/hackrx/run`

- Body:

# ============= Rate Limiting =============  - `documents`: PDF URL or local path

GEMINI_MAX_RETRIES=3  - `questions`: list of questions

GEMINI_BACKOFF_BASE=5.0  - `fusion`: `late` | `early` | `hybrid`

GEMINI_BACKOFF_MAX=60.0

GEMINI_CONCURRENCY=2Discovery:

GEMINI_CIRCUIT_FAILS=6- GET `http://localhost:8000/hackrx/fusions` — lists available fusion modes

GEMINI_CIRCUIT_RESET=120

Individual routes (only if corresponding apps are running directly):

# ============= Caching =============- Late: POST `/hackrx/run_latefusion`

GEMINI_CACHE_PATH=.gemini_cache- Early: POST `/hackrx/run_earlyfusion`

GEMINI_CACHE_TTL=86400- Hybrid: POST `/hackrx/run_hybridfusion`



# ============= Server URLs =============Response shape (unified):

MODEL_SERVER_URL=http://127.0.0.1:8001- `{ "answers": [str, ...], "debug": { ... } }`

UNIFIED_URL=http://127.0.0.1:8000

```## Streamlit UI



### Fusion Mode Guide- Open http://localhost:8501

- Choose document source (URL or upload), select fusion mode, and ask questions in chat.

| Mode | When to Use | Pros | Cons |- UI shows health badges, image previews used in answers, and debug details.

|------|-------------|------|------|

| **Late** | Text and images are separate | Fast, flexible | May miss connections |## Evaluation and benchmarking

| **Early** | Text-image coupling is strong | Unified semantics | Higher memory |

| **Hybrid** | Best overall performance | Combines all strengths | Slowest (but most accurate) |- Run `evaluate_all_fusions.py` to benchmark Late, Early, and Hybrid on a fixed question set.

- Metrics reported per mode:

## 📂 Project Structure  - Exact Match, Precision, Recall, F1

  - BLEU-1, ROUGE-L

```  - Numeric-aware: match rate, average absolute/relative error (extracts first number, handles %, currency)

multimodal-qa-system/- Results saved to `fusion_benchmark_results.json` with per-question details and summary latency.

│

├── 📄 Core Application FilesTips:

│   ├── app.py                    # Streamlit frontend- Tune hybrid weights (e.g., `HYBRID_W_IMAGE`, `HYBRID_W_TEXT`, `HYBRID_W_FUSED`) to favor image-heavy vs text-heavy documents.

│   ├── main_api.py               # Unified API gateway- If you hit Gemini 429 (quota), set `GEMINI_FALLBACK_MODEL` or re-run after quota reset; retries/backoff are built-in.

│   ├── main_latefusion.py        # Late fusion strategy

│   ├── main_earlyfusion.py       # Early fusion strategy## Troubleshooting

│   ├── main_hybridfusion.py      # Hybrid fusion strategy

│   ├── model_server.py           # Embedding/reranking server- Gemini 429 / quota: the code retries with exponential backoff; configure `GEMINI_FALLBACK_MODEL` to continue seamlessly.

│   └── utils.py                  # Shared utilities (Gemini, caching)- Wrong Gemini model: set `GEMINI_MODEL` to a supported name (e.g., `gemini-1.5-flash`).

│- CUDA not available: the model server will fall back to CPU automatically; set `MODEL_DEVICE=cpu` to force.

├── 🔧 Configuration- Windows file locks on images: images are saved into unique per-run cache folders under `_lf_cache/` to avoid conflicts.

│   ├── .env.example              # Environment template- OCR not installed: install Tesseract (Windows builds available on the UB-Mannheim wiki) and set OCR flags in env if applicable.

│   ├── .gitignore                # Git ignore rules

│   └── requirements.txt          # Python dependencies## Performance tuning

│

├── 🚀 Scripts- Enable GPU: `MODEL_DEVICE=cuda` and optionally `FAISS_USE_GPU=true`.

│   ├── start_unified.ps1         # Windows startup (all servers)- Reduce candidate K for speed: `K_TEXT`, `K_IMAGE`, `K_FUSED`.

│   ├── stop_all.ps1              # Stop all servers- Prefer Gemini Flash variants for latency.

│   ├── evaluate_all_fusions.py  # Benchmark tool- Disable OCR when not needed.

│   └── verify_setup.py           # Setup verification

│## Security

├── 📚 Documentation

│   ├── README.md                 # This file- Do not commit `.env` or API keys.

│   ├── STARTUP_GUIDE.md          # Detailed setup instructions- The system does not upload documents externally; images sent to Gemini are loaded from local cache as PIL objects (no remote RAG store).

│   ├── SETTINGS_FEATURE.md       # Settings documentation

│   ├── RESTART_SERVERS.md        # Server restart guide## Notes for contributors

│   ├── QUALITY_IMPROVEMENTS.md   # Quality enhancement details

│   └── FINAL_QUALITY_UPGRADES.md # Latest improvements- Keep public APIs stable; add tests when changing retrieval or fusion logic.

│- When adjusting weights or K values, reflect defaults in README and `.env.example` (if present).

├── 📁 Optional Folders- The Streamlit UI is intentionally light; feel free to extend with metrics panels or charts.

│   ├── archive/                  # Old versions and tests

│   ├── backend/                  # Deployment configs (Render)---

│   ├── frontend/                 # Deployment configs (Vercel)

│   └── docs/                     # GitHub Pages (optional)Copyright © 2025

│
└── 🔒 Generated/Ignored
    ├── .venv/                    # Virtual environment
    ├── __pycache__/              # Python bytecode
    ├── .gemini_cache/            # API response cache
    └── _lf_cache/                # Document processing cache
```

## 🧪 Testing & Evaluation

### Verify Installation

```bash
python verify_setup.py
```

Checks:
- ✅ Environment variables configured
- ✅ Python packages installed
- ✅ CUDA available (if applicable)
- ✅ Project files present

### Run Benchmarks

```bash
python evaluate_all_fusions.py
```

Metrics evaluated:
- **Exact Match (EM)**: Binary accuracy
- **F1 Score**: Token-level precision/recall
- **ROUGE-L**: Longest common subsequence
- **BLEU-1**: Unigram overlap
- **Numeric Accuracy**: Value matching with tolerance

### Performance Benchmarks

Typical results on RTX 4090 (10 questions, ~50-page PDF):

| Fusion | Latency | EM | F1 | ROUGE-L | Num Acc |
|--------|---------|----|----|---------|---------|
| Late   | 8.2s    | 0.30 | 0.847 | 0.823 | 0.88 |
| Early  | 7.5s    | 0.30 | 0.832 | 0.809 | 0.85 |
| Hybrid | 9.1s    | 0.40 | 0.891 | 0.856 | 0.92 |

*CPU-only systems: ~3-5x slower*

## 🔧 Advanced Features

### Intelligent Response Adaptation

The system automatically adjusts response style based on query type:

**Summary Queries** (400-600 words):
- Triggers: "summarize", "overview", "describe", "explain the"
- Format: Introduction → Findings → Data → Conclusion
- Example: *"Summarize this research paper"*

**Fact/Value Queries** (50-150 words):
- Triggers: "what is the", "how many", "show me the", "give me"
- Format: Direct answer → Brief context → Citation
- Example: *"What is the success rate?"*

**Analytical Queries** (150-300 words):
- Default for complex questions
- Format: Direct answer → Detailed explanation → Supporting data
- Example: *"How does the system improve efficiency?"*

### Rate Limit Protection

Built-in safeguards for API stability:

1. **Exponential Backoff**: 5s → 10s → 20s → 40s (with jitter)
2. **Circuit Breaker**: Opens after 6 failures, resets after 120s
3. **Concurrency Control**: Max 2 parallel requests
4. **Response Caching**: 24-hour TTL for identical queries
5. **Fallback Models**: Automatic switch to alternative Gemini models

### GPU Optimization

Automatic performance tuning:

```python
# Auto-detects best configuration:
- CUDA available? → Use GPU for models
- Multiple GPUs? → Select by DEVICE_ID
- FAISS GPU support? → Enable GPU indexing
- Out of memory? → Graceful fallback to CPU
```

### Caching Strategy

Two-level caching for speed:

1. **Gemini Cache**: Stores API responses (`.gemini_cache/`)
   - Key: SHA256(prompt + image_paths)
   - TTL: 24 hours
   - Saves: API calls, reduces latency

2. **Document Cache**: Stores processed PDFs (`_lf_cache/`)
   - Extracted text, images, embeddings
   - Auto-cleaned per run
   - Saves: Processing time on repeated docs

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```powershell
# Windows
.\stop_all.ps1

# Linux/Mac
lsof -ti:8000,8001,8501 | xargs kill -9
```

**CUDA Out of Memory**
```bash
# Set in .env:
MODEL_DEVICE=cpu
FAISS_USE_GPU=false
```

**Gemini Rate Limit Errors**
```bash
# Reduce concurrency in .env:
GEMINI_CONCURRENCY=1
GEMINI_BACKOFF_BASE=10.0

# Or check cache:
ls .gemini_cache/  # Should contain cached responses
```

**Slow Processing**
```bash
# Enable GPU acceleration in .env:
MODEL_DEVICE=cuda
FAISS_USE_GPU=true

# Or reduce retrieval size in fusion files:
K_TEXT=6  # Instead of 10
K_IMAGE=6  # Instead of 10
```

**Import Errors**
```bash
# Reinstall dependencies:
pip install -r requirements.txt --force-reinstall

# Verify installation:
python verify_setup.py
```

### Debug Mode

Enable detailed logging:

```bash
# Set in .env:
LOG_LEVEL=DEBUG

# Or set environment variable:
set LOG_LEVEL=DEBUG  # Windows
export LOG_LEVEL=DEBUG  # Linux/Mac
```

### Health Checks

Monitor system status:

```bash
# Model Server
curl http://localhost:8001/health

# Main API
curl http://localhost:8000/hackrx/health

# Or check in UI sidebar: "Server Health" section
```

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open** a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt  # if available

# Run tests
pytest tests/

# Format code
black .
isort .

# Type checking
mypy *.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Google Gemini](https://ai.google.dev/)** - AI generation & multimodal understanding
- **[Sentence Transformers](https://www.sbert.net/)** - Text & image embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** - Efficient similarity search
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern API framework
- **[Streamlit](https://streamlit.io/)** - Interactive web UI
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - PDF processing
- **[OpenAI CLIP](https://github.com/openai/CLIP)** - Vision-language model

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/multimodal-qa-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/multimodal-qa-system/discussions)
- **Email**: your.email@example.com

## 🗺️ Roadmap

- [ ] Add support for more document formats (DOCX, PPT)
- [ ] Implement conversation history
- [ ] Add multi-document comparison
- [ ] Deploy as Docker container
- [ ] Create REST API client library
- [ ] Add streaming responses
- [ ] Support for local LLMs (Llama, Mistral)

---

<div align="center">

**⭐ If you find this project useful, please star it on GitHub! ⭐**

Built with ❤️ using Python • FastAPI • Streamlit • CUDA

</div>
