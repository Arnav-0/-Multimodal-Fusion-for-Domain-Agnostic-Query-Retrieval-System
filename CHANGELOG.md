# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Docker containerization
- Support for DOCX and PPT formats
- Conversation history feature
- Streaming API responses
- Local LLM support (Llama, Mistral)

## [1.0.0] - 2025-11-02

### Added
- **Three Fusion Strategies**: Late, Early, and Hybrid fusion modes for optimal retrieval
- **Intelligent Response Sizing**: Automatically adjusts answer length based on question type
  - Summary queries: 400-600 words with comprehensive analysis
  - Fact/value queries: 50-150 words with direct answers
  - Analytical queries: 150-300 words with balanced detail
- **Settings Panel**: Configure API URL directly from UI without code changes
- **GPU Acceleration**: Full CUDA support for embeddings, FAISS indexing, and model inference
- **Rate Limit Protection**: Exponential backoff, circuit breaker, and response caching
- **Multimodal Processing**:
  - Advanced text extraction with block-based processing
  - OCR for tables with structured mode (--psm 6)
  - CLIP-based image understanding
  - Visual content analysis (tables, charts, graphs)
- **Comprehensive Evaluation**: Benchmark script with EM, F1, ROUGE-L, BLEU-1 metrics
- **Interactive UI**: Modern Streamlit dashboard with real-time feedback
- **Health Monitoring**: Live server status display in UI
- **Document Caching**: Two-level caching for API responses and processed documents

### Changed
- **Improved Prompts**: Context-aware prompts with explicit instructions for data extraction
- **Enhanced Retrieval**: Increased context from 1800 to 3000 chars, 8 to 10 pages/images
- **Better Error Handling**: Comprehensive error messages and graceful fallbacks
- **Optimized Reranking**: Increased candidates from 6 to 10-15 for better accuracy

### Fixed
- Gemini API rate limit errors with robust retry logic
- Frontend 404 errors with unified API endpoint
- Type errors in debug tab display
- Fragmented summaries lacking synthesis
- Missing table and graph data in responses

## [0.9.0] - 2025-10-21

### Added
- Late fusion implementation with separate text/image retrieval
- Early fusion with combined embeddings
- Hybrid fusion combining multiple retrieval strategies
- Model server for centralized embedding generation
- Basic Streamlit UI
- PowerShell startup scripts for Windows
- Environment configuration via .env

### Initial Features
- PDF document processing
- Text embedding with E5 model
- Image embedding with CLIP
- FAISS vector search
- Gemini API integration
- Basic Q&A functionality

## Development Milestones

### Phase 1: Core Implementation (Oct 2025)
- [x] PDF text extraction
- [x] Image extraction and processing
- [x] Embedding generation
- [x] Vector search with FAISS
- [x] Basic Q&A with Gemini

### Phase 2: Multi-Modal Enhancement (Oct 2025)
- [x] CLIP image embeddings
- [x] CrossEncoder reranking
- [x] OCR for tables
- [x] Three fusion strategies

### Phase 3: Quality Improvements (Oct-Nov 2025)
- [x] Rate limit handling
- [x] Response caching
- [x] Intelligent prompts
- [x] Context-aware sizing
- [x] Comprehensive evaluation

### Phase 4: Production Ready (Nov 2025)
- [x] Settings UI
- [x] Health monitoring
- [x] Error handling
- [x] Documentation
- [x] GitHub preparation

---

## Release Notes Format

Each release includes:
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

## Contributors

Thank you to all contributors who have helped make this project better!

- [List contributors here]

## Support

For questions or issues:
- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/multimodal-qa-system/issues)
- **Discussions**: [Ask questions](https://github.com/yourusername/multimodal-qa-system/discussions)
