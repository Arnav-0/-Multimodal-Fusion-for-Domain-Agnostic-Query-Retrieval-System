import os
import time
import logging
import asyncio
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple

import requests
import numpy as np

# ---- FAISS (GPU optional) ----
FAISS_USE_GPU = os.getenv("FAISS_USE_GPU", "false").lower() == "true"
import faiss
try:
    import faiss.contrib.torch_utils  # enable GPU interop if available
    FAISS_HAS_GPU = faiss.get_num_gpus() > 0
except Exception:
    FAISS_HAS_GPU = False

import fitz  # PyMuPDF
from PIL import Image
try:
    import pytesseract
except Exception:
    pytesseract = None

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import (
    get_text_embeddings,
    get_clip_text_embeddings,
    get_image_embeddings,
    rerank_candidates,
    gemini_generate,
)

# ---------------- Logging ----------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("latefusion")

# ---------------- App ----------------
app = FastAPI(title="Late Fusion QA (Gemini + FAISS)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---------------- Config ----------------
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "3000"))  # Increased for better context quality
K_TEXT  = int(os.getenv("K_TEXT", "10"))  # Retrieve more pages for comprehensive answers
K_IMAGE = int(os.getenv("K_IMAGE", "10"))  # Retrieve more images/tables/charts
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"

# ---------------- Schemas ----------------
class LFRequest(BaseModel):
    documents: str = Field(..., description="PDF path, directory with PDFs, or http/https URL")
    questions: List[str]
    fusion: str = Field("late", description="late fusion (server supports late only)")

class LFResponse(BaseModel):
    answers: List[str]
    debug: Dict[str, Any]


# ---------------- Helpers ----------------
def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False

def _download_pdf_to(path_or_url: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    if not _is_url(path_or_url):
        if not os.path.exists(path_or_url):
            raise RuntimeError(f"Document not found: {path_or_url}")
        return path_or_url

    filename = os.path.basename(urlparse(path_or_url).path) or "download.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    local_path = os.path.join(dest_dir, filename)

    with requests.get(path_or_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if os.path.getsize(local_path) < 1024:
        raise RuntimeError("Downloaded file seems too small to be a valid PDF")
    return local_path


def _maybe_gpu_index(cpu_index: faiss.Index) -> faiss.Index:
    if FAISS_USE_GPU and FAISS_HAS_GPU:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return cpu_index

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    faiss.normalize_L2(vectors)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(vectors.astype("float32"))
    return _maybe_gpu_index(cpu_index)

def top_k(index: faiss.Index, qvec: np.ndarray, k: int):
    q = qvec.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D[0], I[0]

def extract_text_and_images(pdf_path: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages_text: List[str] = []
    image_paths: List[str] = []
    ocr_snippets: List[str] = []
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            
            # Extract text with better structure preservation
            # Try "blocks" format first for better layout preservation
            try:
                blocks = page.get_text("blocks")
                text_parts = []
                for block in blocks:
                    if len(block) >= 5:  # block format: (x0, y0, x1, y1, "text", block_no, block_type)
                        block_text = block[4].strip()
                        if block_text:
                            text_parts.append(block_text)
                text = "\n".join(text_parts)
            except Exception:
                # Fallback to simple text extraction
                text = page.get_text("text") or ""
            
            # Clean up excessive whitespace but preserve paragraph breaks
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
            
            pages_text.append(f"[Page {page_idx + 1}]\n{text}")

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Skip very small images (likely icons/logos)
                    if pix.width < 50 or pix.height < 50:
                        pix = None
                        continue
                    
                    if pix.n >= 5:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    img_path = os.path.join(out_dir, f"page{page_idx + 1}_img{img_index + 1}.png")
                    pix.save(img_path)
                    pix = None
                    image_paths.append(img_path)

                    if ENABLE_OCR and pytesseract is not None:
                        try:
                            # Enhanced OCR for tables and structured data
                            img_pil = Image.open(img_path)
                            
                            # Try structured OCR first (better for tables)
                            try:
                                txt = pytesseract.image_to_string(img_pil, config='--psm 6') or ""
                            except Exception:
                                # Fallback to default OCR
                                txt = pytesseract.image_to_string(img_pil) or ""
                            
                            # Preserve line breaks for table structure
                            txt = "\n".join(line.strip() for line in txt.split("\n") if line.strip())
                            
                            if txt and len(txt) > 10:  # Only keep meaningful OCR results
                                # Detect if this looks like a table/chart
                                is_table_like = any(indicator in txt.lower() for indicator in 
                                    ["table", "figure", "chart", "graph", "|", "---", "row", "column"])
                                label = "Table/Chart" if is_table_like else "Figure"
                                ocr_snippets.append(f"[Page {page_idx + 1} {label} {img_index + 1}]\n{txt}")
                        except Exception:
                            pass
                except Exception:
                    continue
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return {
        "pages_text": pages_text,
        "image_paths": image_paths,
        "ocr_snippets": ocr_snippets,
    }


# ---------------- Core QA ----------------
async def answer_one_question(query: str, corpus: Dict[str, Any]):
    # Build text index
    if "text_index" not in corpus:
        text_emb = np.array(get_text_embeddings(corpus["pages_text"]), dtype=np.float32)
        corpus["text_emb"] = text_emb
        corpus["text_index"] = build_faiss_index(text_emb)

    # Build image index (if any)
    if "image_index" not in corpus:
        if corpus["image_paths"]:
            img_emb = np.array(get_image_embeddings(corpus["image_paths"]), dtype=np.float32)
            corpus["img_emb"] = img_emb
            corpus["image_index"] = build_faiss_index(img_emb)
        else:
            corpus["image_index"] = None

    # --- Text retrieval ---
    # Detect query type and adjust retrieval parameters
    is_summary = any(word in query.lower() for word in ["summarize", "summary", "overview", "main points", "key findings"])
    
    # For summaries, retrieve more pages and more text per page
    k_text_to_use = min(K_TEXT * 2, len(corpus["pages_text"])) if is_summary else min(K_TEXT, len(corpus["pages_text"]))
    max_chars_to_use = MAX_TEXT_CHARS * 2 if is_summary else MAX_TEXT_CHARS
    
    q_text = np.array(get_text_embeddings([query]), dtype=np.float32)[0]
    D_t, I_t = top_k(corpus["text_index"], q_text, k_text_to_use)
    text_cands = [corpus["pages_text"][i][:max_chars_to_use] for i in I_t]

    # --- Image retrieval (CLIP text->image) ---
    image_cands: List[str] = []
    support_image_paths: List[str] = []
    if corpus["image_index"] is not None and corpus["image_paths"]:
        q_clip = np.array(get_clip_text_embeddings([query]), dtype=np.float32)[0]
        D_i, I_i = top_k(corpus["image_index"], q_clip, min(K_IMAGE, len(corpus["image_paths"])))
        for idx in I_i:
            img_path = corpus["image_paths"][idx]
            support_image_paths.append(img_path)
            base = os.path.basename(img_path).split(".")[0]
            matched_ocr = next((s for s in corpus["ocr_snippets"] if base in s), None)
            image_cands.append(matched_ocr or f"[Figure match] {base}")

    # --- Merge + rerank (cross-encoder) ---
    candidates = text_cands + image_cands
    if candidates:
        scores = rerank_candidates(query, candidates)
        # Use more candidates for better answer quality
        # Summaries need even more context
        num_support = 15 if is_summary else 10
        order = np.argsort(-np.array(scores))[:num_support]
        support_text = [candidates[i] for i in order]
        top_scores = [float(scores[i]) for i in order]
    else:
        support_text, top_scores = text_cands[:8], []

    # Enhance prompt based on query type - INTELLIGENT RESPONSE SIZING
    query_lower = query.lower()
    
    # Detect question type
    is_summary = any(word in query_lower for word in ["summarize", "summary", "overview", "main points", "key findings", "describe", "explain the", "what is"])
    
    # Simple fact/value queries - should be concise
    is_simple_query = any(word in query_lower for word in [
        "what is the", "how many", "how much", "when did", "when was", 
        "who is", "who are", "which", "where", "give me the", "show me the",
        "what's the", "list the", "name the"
    ])
    
    # Data extraction queries - need values with brief context
    is_data_query = any(word in query_lower for word in [
        "value", "number", "percentage", "metric", "score", "count", 
        "total", "amount", "rate", "figure", "statistics"
    ])
    
    if is_summary:
        prompt = (
            "You are an expert technical analyst tasked with creating an EXTREMELY DETAILED and COMPREHENSIVE analysis.\n\n"
            f"USER QUESTION: {query}\n\n"
            "📚 DOCUMENT CONTENT (Text, Tables, Charts, Figures):\n\n" + 
            "\n\n" + "="*80 + "\n\n".join(f"\n\n📄 SECTION {i+1}:\n{s}\n\n{'='*80}" for i, s in enumerate(support_text)) +
            "\n\n🎯 YOUR MISSION - CRITICAL REQUIREMENTS:\n\n"
            "⚠️ **ABSOLUTELY MANDATORY - READ CAREFULLY:**\n\n"
            "1️⃣ **LENGTH REQUIREMENT:** Your response MUST be AT LEAST 400-600 words. SHORT ANSWERS WILL BE REJECTED.\n\n"
            "2️⃣ **COMPREHENSIVE COVERAGE:** Discuss EVERY major topic, component, feature, and finding mentioned across ALL sections above.\n\n"
            "3️⃣ **EXTRACT ALL DATA:** Include EVERY number, percentage, metric, statistic, measurement, and quantitative value you find. Don't skip any data points.\n\n"
            "4️⃣ **TABLE/CHART ANALYSIS:** If any tables, charts, graphs, or figures are mentioned:\n"
            "   - Describe what they show in detail\n"
            "   - List all data points, values, and trends\n"
            "   - Explain the significance of the visual data\n\n"
            "5️⃣ **LOGICAL STRUCTURE:** Organize with clear sections:\n"
            "   - Introduction/Overview (what this is about)\n"
            "   - Main Content (multiple detailed paragraphs covering different aspects)\n"
            "   - Key Components/Features (bulleted lists with explanations)\n"
            "   - Technical Details/Methodology (if applicable)\n"
            "   - Results/Findings/Metrics (with all numbers)\n"
            "   - Conclusion/Summary\n\n"
            "6️⃣ **SYNTHESIS:** Connect information across sections - show relationships and build a cohesive narrative.\n\n"
            "7️⃣ **SPECIFIC CITATIONS:** Cite page numbers like 'Page 5' or 'Pages 3-4' for every fact.\n\n"
            "8️⃣ **PROFESSIONAL TONE:** Write in clear, detailed paragraphs. Use subheadings. Make it thorough and informative.\n\n"
            "❌ **DO NOT:**\n"
            "- Give short, superficial summaries\n"
            "- Skip sections or details\n"
            "- Omit numbers, percentages, or data\n"
            "- Use only citations without explaining content\n"
            "- Write less than 400 words\n\n"
            "✅ **BEGIN YOUR DETAILED ANALYSIS NOW:**"
        )
    elif is_simple_query or is_data_query:
        # For simple fact/value queries - concise but complete
        prompt = (
            "You are an expert analyst providing ACCURATE, CONCISE answers.\n\n"
            f"USER QUESTION: {query}\n\n"
            "📚 AVAILABLE INFORMATION:\n\n" + 
            "\n\n".join(f"📄 SOURCE {i+1}:\n{s}" for i, s in enumerate(support_text)) +
            "\n\n🎯 ANSWER FORMAT:\n\n"
            "**IMPORTANT:** This is a SPECIFIC VALUE/FACT QUERY. Provide a FOCUSED, CONCISE response.\n\n"
            "✅ **DO THIS:**\n"
            "1. **State the answer FIRST** - Give the direct answer in the first sentence\n"
            "2. **Provide brief context** - 2-3 sentences explaining what this value represents\n"
            "3. **Include relevant details** - Any important related information (50-150 words total)\n"
            "4. **Cite the source** - Mention which page(s) contain this information\n"
            "5. **Use proper formatting:**\n"
            "   - For single values: Bold the key value/answer\n"
            "   - For lists: Use bullet points\n"
            "   - For tables: Use structured format\n\n"
            "❌ **DO NOT:**\n"
            "- Write long essays (keep it under 150 words unless multiple values requested)\n"
            "- Include unnecessary background information\n"
            "- Repeat the question in your answer\n"
            "- Be vague - use EXACT values, names, dates, numbers\n\n"
            "📋 **EXAMPLE FORMAT:**\n"
            "For 'What is the success rate?':\n"
            "✓ GOOD: 'The success rate is **87.3%** (Page 5). This represents the percentage of projects completed on time and within budget during Q3 2024.'\n"
            "✗ BAD: 'According to the document, various metrics were tracked throughout the project lifecycle. The success rate, which is an important indicator of performance...'\n\n"
            "✅ **PROVIDE YOUR FOCUSED ANSWER NOW:**"
        )
    else:
        # For explanatory/analytical questions - balanced detail
        prompt = (
            "You are an expert analyst providing CLEAR, WELL-STRUCTURED answers.\n\n"
            f"USER QUESTION: {query}\n\n"
            "📚 AVAILABLE INFORMATION (Text, Tables, Charts, Figures):\n\n" + 
            "\n\n" + "="*80 + "\n\n".join(f"\n\n📄 SOURCE {i+1}:\n{s}\n\n{'='*80}" for i, s in enumerate(support_text)) +
            "\n\n🎯 ANSWER REQUIREMENTS:\n\n"
            "**Response Length:** Aim for 150-300 words (adjust based on question complexity)\n\n"
            "✅ **STRUCTURE YOUR ANSWER:**\n\n"
            "1️⃣ **DIRECT ANSWER FIRST:** Start with the key point/answer in the first 1-2 sentences\n\n"
            "2️⃣ **EXPLAIN WITH DETAIL:** Provide thorough explanation with:\n"
            "   - All relevant numbers, percentages, metrics\n"
            "   - Context and significance\n"
            "   - Relationships between concepts\n\n"
            "3️⃣ **TABLES/CHARTS:** If visual data is mentioned:\n"
            "   - Describe what it shows\n"
            "   - List key data points\n"
            "   - Explain trends\n\n"
            "4️⃣ **BE SPECIFIC:** Use exact figures, not vague terms ('87.3%' not 'high percentage')\n\n"
            "5️⃣ **CITE SOURCES:** Reference page numbers for facts (e.g., 'According to Page 5...')\n\n"
            "6️⃣ **CLEAR FORMATTING:**\n"
            "   - Use paragraphs for explanations\n"
            "   - Use bullet points for lists\n"
            "   - Bold important values/terms\n"
            "   - Add subheadings if covering multiple aspects\n\n"
            "❌ **AVOID:**\n"
            "- Overly long responses for simple questions\n"
            "- Vague statements without data\n"
            "- Skipping relevant information\n"
            "- Poor formatting (walls of text)\n\n"
            "✅ **PROVIDE YOUR CLEAR, BALANCED ANSWER:**"
        )

    # Use images only if enabled in corpus
    images_for_prompt = support_image_paths[:3] if corpus.get("include_images", True) else []
    answer = gemini_generate(prompt, image_paths=images_for_prompt)
    debug = {
        "text_hits": len(text_cands),
        "image_hits": len(image_cands),
        "used_images": support_image_paths[:3],
        "top_scores": top_scores,
    }
    return answer, debug


# ---------------- Routes ----------------
@app.post("/hackrx/run_latefusion", response_model=LFResponse)
async def run_latefusion(body: LFRequest, include_images: bool = True):
    input_path = body.documents

    cache_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "_lf_cache")
    os.makedirs(cache_root, exist_ok=True)
    run_dir = os.path.join(cache_root, f"late_{int(time.time()*1000)}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        if _is_url(input_path):
            pdf_path = _download_pdf_to(input_path, cache_root)
        elif os.path.isdir(input_path):
            pdfs = [p for p in os.listdir(input_path) if p.lower().endswith(".pdf")]
            if not pdfs:
                return LFResponse(answers=["[Error: no PDF found in directory]"], debug={"dir": input_path})
            pdf_path = os.path.join(input_path, pdfs[0])
        else:
            if not os.path.exists(input_path):
                return LFResponse(answers=["[Error: document path not found]"], debug={"path": input_path})
            pdf_path = input_path
    except Exception as e:
        return LFResponse(answers=[f"[Error: {e}]"], debug={"path": input_path})

    parsed = extract_text_and_images(pdf_path, run_dir)
    corpus = {
        "pages_text": parsed["pages_text"],
        "image_paths": parsed["image_paths"],
        "ocr_snippets": parsed["ocr_snippets"],
        "include_images": include_images,
    }

    tasks = [answer_one_question(q, corpus) for q in body.questions]
    results = await asyncio.gather(*tasks)
    answers = [r[0] for r in results]
    debug = {
        "per_q": [r[1] for r in results],
        "doc": os.path.basename(pdf_path),
        "pages": len(corpus["pages_text"]),
        "images": len(corpus["image_paths"]),
        "fusion": body.fusion,
        "faiss_gpu": FAISS_USE_GPU and FAISS_HAS_GPU,
        "include_images": include_images,
    }
    return LFResponse(answers=answers, debug=debug)
