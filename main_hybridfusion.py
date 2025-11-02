import os
import time
import logging
import asyncio
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, DefaultDict, Tuple
from collections import defaultdict

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
logger = logging.getLogger("hybridfusion")

# ---------------- App ----------------
app = FastAPI(title="Hybrid Fusion QA (Gemini + FAISS)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---------------- Config ----------------
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "3000"))  # Increased for better quality
K_TEXT = int(os.getenv("K_TEXT", "10"))  # More text retrieval
K_IMAGE = int(os.getenv("K_IMAGE", "10"))  # More image retrieval
K_FUSED = int(os.getenv("K_FUSED", "10"))  # More fused retrieval
MAX_SUPPORT_PAGES = int(os.getenv("MAX_SUPPORT_PAGES", "10"))  # More pages in final context
MAX_SUPPORT_IMAGES = int(os.getenv("MAX_SUPPORT_IMAGES", "5"))  # More images/tables/charts
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"

# Early-fusion style weights for building fused vectors
FUSION_TEXT_WEIGHT = float(os.getenv("FUSION_TEXT_WEIGHT", "1.0"))
FUSION_IMAGE_WEIGHT = float(os.getenv("FUSION_IMAGE_WEIGHT", "1.0"))

# Hybrid combiner weights for score blending
HYBRID_W_FUSED = float(os.getenv("HYBRID_W_FUSED", "1.0"))
HYBRID_W_TEXT = float(os.getenv("HYBRID_W_TEXT", "0.7"))
HYBRID_W_IMAGE = float(os.getenv("HYBRID_W_IMAGE", "0.7"))


# ---------------- Schemas ----------------
class HFRequest(BaseModel):
    documents: str = Field(..., description="PDF path, directory with PDFs, or http/https URL")
    questions: List[str]
    fusion: str = Field("hybrid", description="hybrid fusion")


class HFResponse(BaseModel):
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
            text = page.get_text("text") or ""
            text = " ".join(text.split())
            pages_text.append(f"[Page {page_idx + 1}] {text}")

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_path = os.path.join(out_dir, f"page{page_idx + 1}_img{img_index + 1}.png")
                pix.save(img_path)
                pix = None
                image_paths.append(img_path)

                if ENABLE_OCR and pytesseract is not None:
                    try:
                        txt = pytesseract.image_to_string(Image.open(img_path)) or ""
                        txt = " ".join(txt.split())
                        if txt:
                            ocr_snippets.append(f"[Page {page_idx + 1} Fig {img_index + 1}] {txt}")
                    except Exception:
                        pass
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


def _page_from_img_path(p: str) -> Optional[int]:
    try:
        base = os.path.basename(p)
        if base.startswith("page") and "_img" in base:
            num = base.split("_img")[0].replace("page", "")
            return int(num) - 1
    except Exception:
        return None
    return None


def build_hybrid_corpus(parsed: Dict[str, Any]) -> Dict[str, Any]:
    pages_text: List[str] = parsed["pages_text"]
    image_paths: List[str] = parsed["image_paths"]
    ocr_snippets: List[str] = parsed["ocr_snippets"]

    # Text embeddings per page
    text_emb = np.array(get_text_embeddings(pages_text), dtype=np.float32)
    if text_emb.size == 0:
        text_emb = np.zeros((len(pages_text), 768), dtype=np.float32)

    # Image embeddings per image
    if image_paths:
        img_emb_all = np.array(get_image_embeddings(image_paths), dtype=np.float32)
    else:
        img_emb_all = np.zeros((0, 768), dtype=np.float32)

    # Page -> image indices mapping and per-page averaged image vectors
    page_to_img_indices: DefaultDict[int, List[int]] = defaultdict(list)
    for idx, p in enumerate(image_paths):
        pi = _page_from_img_path(p)
        if pi is not None and 0 <= pi < len(pages_text):
            page_to_img_indices[pi].append(idx)

    # Determine clip dimension
    clip_probe = np.array(get_clip_text_embeddings(["probe"]), dtype=np.float32)
    clip_dim = clip_probe.shape[1] if clip_probe.size > 0 else (img_emb_all.shape[1] if img_emb_all.size > 0 else 768)

    page_img_vecs = np.zeros((len(pages_text), clip_dim), dtype=np.float32)
    for p_idx in range(len(pages_text)):
        ids = page_to_img_indices.get(p_idx, [])
        if ids and img_emb_all.size > 0:
            page_img_vecs[p_idx] = np.mean(img_emb_all[ids, :], axis=0)
        else:
            page_img_vecs[p_idx] = np.zeros((clip_dim,), dtype=np.float32)

    # Fused vectors
    fused_pages = np.concatenate([
        text_emb * FUSION_TEXT_WEIGHT,
        page_img_vecs * FUSION_IMAGE_WEIGHT,
    ], axis=1)

    # Build indices
    text_index = build_faiss_index(text_emb)
    fused_index = build_faiss_index(fused_pages)
    image_index = build_faiss_index(img_emb_all) if img_emb_all.size > 0 else None

    return {
        "pages_text": pages_text,
        "image_paths": image_paths,
        "ocr_snippets": ocr_snippets,
        "text_emb": text_emb,
        "page_img_vecs": page_img_vecs,
        "img_emb_all": img_emb_all,
        "indices": {
            "text": text_index,
            "fused": fused_index,
            "image": image_index,
        },
        "clip_dim": clip_dim,
        "page_to_img_indices": page_to_img_indices,
    }


def build_query_vecs(query: str, clip_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_t = np.array(get_text_embeddings([query]), dtype=np.float32)
    q_c = np.array(get_clip_text_embeddings([query]), dtype=np.float32)
    if q_t.size == 0:
        q_t = np.zeros((1, 768), dtype=np.float32)
    if q_c.size == 0:
        q_c = np.zeros((1, clip_dim), dtype=np.float32)
    q_fused = np.concatenate([
        q_t[0] * FUSION_TEXT_WEIGHT,
        q_c[0] * FUSION_IMAGE_WEIGHT,
    ], axis=0)
    return q_t[0], q_c[0], q_fused.astype(np.float32)


def _combine_page_scores(
    fused_hits: Tuple[np.ndarray, np.ndarray],
    text_hits: Tuple[np.ndarray, np.ndarray],
    image_hits: Optional[Tuple[np.ndarray, np.ndarray]],
    image_paths: List[str],
) -> Tuple[List[int], Dict[int, float], Dict[int, List[int]]]:
    # fused_hits: (D_f, I_f) over pages
    # text_hits:  (D_t, I_t) over pages
    # image_hits: (D_i, I_i) over images
    # returns: sorted_pages, page_score_map, page->imageIndices map (for top-N images)
    page_scores: Dict[int, float] = defaultdict(float)
    page_img_map: Dict[int, List[int]] = defaultdict(list)

    Df, If = fused_hits
    for s, idx in zip(Df.tolist(), If.tolist()):
        page_scores[int(idx)] += HYBRID_W_FUSED * float(s)

    Dt, It = text_hits
    for s, idx in zip(Dt.tolist(), It.tolist()):
        page_scores[int(idx)] += HYBRID_W_TEXT * float(s)

    if image_hits is not None:
        Di, Ii = image_hits
        for s, img_idx in zip(Di.tolist(), Ii.tolist()):
            # map image to page
            p = _page_from_img_path(image_paths[int(img_idx)])
            if p is not None:
                page_scores[int(p)] += HYBRID_W_IMAGE * float(s)
                page_img_map[int(p)].append(int(img_idx))

    sorted_pages = sorted(page_scores.keys(), key=lambda p: page_scores[p], reverse=True)
    return sorted_pages, page_scores, page_img_map


async def answer_one_question_hybrid(query: str, corpus: Dict[str, Any]):
    indices = corpus["indices"]
    q_t, q_c, q_f = build_query_vecs(query, corpus["clip_dim"])

    # Retrieve independently
    Df, If = top_k(indices["fused"], q_f, min(K_FUSED, len(corpus["pages_text"])))
    Dt, It = top_k(indices["text"], q_t, min(K_TEXT, len(corpus["pages_text"])))

    Di, Ii = (None, None)
    if indices.get("image") is not None and len(corpus["image_paths"]) > 0:
        Di, Ii = top_k(indices["image"], q_c, min(K_IMAGE, len(corpus["image_paths"])) )

    # Combine page scores
    sorted_pages, page_scores, page_img_map = _combine_page_scores((Df, If), (Dt, It),
                                                                   (Di, Ii) if Di is not None else None,
                                                                   corpus["image_paths"]) 

    # Build supports
    support_pages = sorted_pages[:MAX_SUPPORT_PAGES]
    support_texts: List[str] = []
    support_images: List[str] = []

    # quick OCR lookup by page tag
    for p in support_pages:
        text = corpus["pages_text"][p][:MAX_TEXT_CHARS]
        support_texts.append(text)

        page_tag = f"[Page {p + 1}]"
        page_ocr = [s for s in corpus["ocr_snippets"] if page_tag in s]
        if page_ocr:
            support_texts.append(page_ocr[0])

        # add top images tied to this page (if any were part of image retrieval)
        for img_idx in page_img_map.get(p, [])[:2]:
            path = corpus["image_paths"][img_idx]
            if len(support_images) < MAX_SUPPORT_IMAGES and path not in support_images:
                support_images.append(path)

    # If still have image slots, add globally top images
    if len(support_images) < MAX_SUPPORT_IMAGES and Di is not None:
        for img_idx in Ii.tolist():
            pth = corpus["image_paths"][int(img_idx)]
            if pth not in support_images:
                support_images.append(pth)
                if len(support_images) >= MAX_SUPPORT_IMAGES:
                    break

    # Optional textual rerank
    rerank_scores: List[float] = []
    if support_texts:
        rerank_scores = rerank_candidates(query, support_texts)
        order = np.argsort(-np.array(rerank_scores))[:MAX_SUPPORT_PAGES]
        support_texts = [support_texts[i] for i in order]
        rerank_scores = [float(rerank_scores[i]) for i in order]

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
            "\n\n" + "="*80 + "\n\n".join(f"\n\n📄 SECTION {i+1}:\n{s}\n\n{'='*80}" for i, s in enumerate(support_texts)) +
            "\n\n⚠️ **ABSOLUTELY MANDATORY - READ CAREFULLY:**\n\n"
            "1️⃣ **LENGTH REQUIREMENT:** Your response MUST be AT LEAST 400-600 words. SHORT ANSWERS WILL BE REJECTED.\n\n"
            "2️⃣ **SYNTHESIZE EVERYTHING:** Create a cohesive, flowing narrative that connects information from ALL sections above. Do NOT just list facts from individual sections.\n\n"
            "3️⃣ **EXTRACT ALL DATA:** Include EVERY relevant number, percentage, statistic, metric from tables and text. List them explicitly.\n\n"
            "4️⃣ **ANALYZE VISUAL CONTENT:** For every table, chart, or graph mentioned:\n"
            "   - Describe its purpose and what it shows\n"
            "   - List specific data points and values\n"
            "   - Explain trends, patterns, or insights\n\n"
            "5️⃣ **STRUCTURED FORMAT:** Use clear sections:\n"
            "   - Introduction/Overview\n"
            "   - Detailed Findings (with subheadings if needed)\n"
            "   - Key Data & Metrics\n"
            "   - Conclusion/Summary\n\n"
            "6️⃣ **BE SPECIFIC:** Use exact figures, not vague terms. Say '87.3%' not 'high percentage'. Say '23 customers' not 'many customers'.\n\n"
            "7️⃣ **CITE SOURCES:** Reference page numbers for all facts (e.g., 'According to Page 5...', 'As shown on Page 8...')\n\n"
            "8️⃣ **PROFESSIONAL DEPTH:** Write detailed paragraphs demonstrating deep understanding. Explain WHY data matters, not just WHAT it says.\n\n"
            "❌ **DO NOT:**\n"
            "- Provide short, superficial summaries\n"
            "- Skip numerical data or table content\n"
            "- Use vague language or generalizations\n"
            "- Write less than 400 words\n\n"
            "✅ **BEGIN YOUR DETAILED ANALYSIS NOW:**"
        )
    elif is_simple_query or is_data_query:
        # For simple fact/value queries - concise but complete
        prompt = (
            "You are an expert analyst providing ACCURATE, CONCISE answers.\n\n"
            f"USER QUESTION: {query}\n\n"
            "📚 AVAILABLE INFORMATION:\n\n" + 
            "\n\n".join(f"📄 SOURCE {i+1}:\n{s}" for i, s in enumerate(support_texts)) +
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
            "\n\n" + "="*80 + "\n\n".join(f"\n\n📄 SOURCE {i+1}:\n{s}\n\n{'='*80}" for i, s in enumerate(support_texts)) +
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

    # Use images only if include_images is enabled
    images_to_use = support_images[:MAX_SUPPORT_IMAGES] if corpus.get("include_images", True) else []
    answer = gemini_generate(prompt, image_paths=images_to_use)
    debug = {
        "pages_considered": [int(p) + 1 for p in support_pages],
        "page_scores": {str(int(p)+1): float(page_scores[p]) for p in support_pages},
        "used_images": support_images[:MAX_SUPPORT_IMAGES],
        "fused_top": [int(i)+1 for i in If.tolist()],
        "text_top": [int(i)+1 for i in It.tolist()],
        "image_top": [int(i) for i in Ii.tolist()] if (Ii is not None) else [],
        "rerank": rerank_scores,
    }
    return answer, debug


# ---------------- Routes ----------------
class _HFBody(HFRequest):
    pass


@app.post("/hackrx/run_hybridfusion", response_model=HFResponse)
async def run_hybridfusion(body: _HFBody, include_images: bool = True):
    input_path = body.documents

    cache_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "_lf_cache")
    os.makedirs(cache_root, exist_ok=True)
    run_dir = os.path.join(cache_root, f"hybrid_{int(time.time()*1000)}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        if _is_url(input_path):
            pdf_path = _download_pdf_to(input_path, cache_root)
        elif os.path.isdir(input_path):
            pdfs = [p for p in os.listdir(input_path) if p.lower().endswith(".pdf")]
            if not pdfs:
                return HFResponse(answers=["[Error: no PDF found in directory]"] , debug={"dir": input_path})
            pdf_path = os.path.join(input_path, pdfs[0])
        else:
            if not os.path.exists(input_path):
                return HFResponse(answers=["[Error: document path not found]"], debug={"path": input_path})
            pdf_path = input_path
    except Exception as e:
        return HFResponse(answers=[f"[Error: {e}]"], debug={"path": input_path})

    parsed = extract_text_and_images(pdf_path, run_dir)
    corpus = build_hybrid_corpus(parsed)
    corpus["include_images"] = include_images

    tasks = [answer_one_question_hybrid(q, corpus) for q in body.questions]
    results = await asyncio.gather(*tasks)
    answers = [r[0] for r in results]
    debug = {
        "per_q": [r[1] for r in results],
        "doc": os.path.basename(pdf_path),
        "pages": len(corpus["pages_text"]),
        "images": len(corpus["image_paths"]),
        "fusion": body.fusion,
        "faiss_gpu": FAISS_USE_GPU and FAISS_HAS_GPU,
        "weights": {
            "fusion_text": FUSION_TEXT_WEIGHT,
            "fusion_image": FUSION_IMAGE_WEIGHT,
            "hybrid_fused": HYBRID_W_FUSED,
            "hybrid_text": HYBRID_W_TEXT,
            "hybrid_image": HYBRID_W_IMAGE,
        },
        "include_images": include_images,
    }
    return HFResponse(answers=answers, debug=debug)
