import os
import time
import logging
import asyncio
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple, Optional, DefaultDict
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
logger = logging.getLogger("earlyfusion")

# ---------------- App ----------------
app = FastAPI(title="Early Fusion QA (Gemini + FAISS)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---------------- Config ----------------
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "3000"))  # Increased for better quality
K_PAGES = int(os.getenv("K_PAGES", os.getenv("K_TEXT", "10")))  # Retrieve more pages
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"

# Early fusion weights
FUSION_TEXT_WEIGHT = float(os.getenv("FUSION_TEXT_WEIGHT", "1.0"))
FUSION_IMAGE_WEIGHT = float(os.getenv("FUSION_IMAGE_WEIGHT", "1.0"))


# ---------------- Schemas ----------------
class EFRequest(BaseModel):
    documents: str = Field(..., description="PDF path, directory with PDFs, or http/https URL")
    questions: List[str]
    fusion: str = Field("early", description="early fusion")


class EFResponse(BaseModel):
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


# ---------------- Early Fusion Core ----------------
def _page_from_img_path(p: str) -> Optional[int]:
    # expects filename like page{n}_img{m}.png
    try:
        base = os.path.basename(p)
        if base.startswith("page") and "_img" in base:
            num = base.split("_img")[0].replace("page", "")
            return int(num) - 1  # zero-based
    except Exception:
        return None
    return None


def build_early_fusion_corpus(parsed: Dict[str, Any]) -> Dict[str, Any]:
    pages_text: List[str] = parsed["pages_text"]
    image_paths: List[str] = parsed["image_paths"]
    ocr_snippets: List[str] = parsed["ocr_snippets"]

    # --- Embed text per page ---
    text_emb = np.array(get_text_embeddings(pages_text), dtype=np.float32)
    if text_emb.size == 0:
        # fallback to zeros to keep shapes valid
        text_emb = np.zeros((len(pages_text), 768), dtype=np.float32)

    # --- Map images to pages ---
    page_to_img_indices: DefaultDict[int, List[int]] = defaultdict(list)
    for idx, p in enumerate(image_paths):
        page_idx = _page_from_img_path(p)
        if page_idx is not None and 0 <= page_idx < len(pages_text):
            page_to_img_indices[page_idx].append(idx)

    # --- Embed images once ---
    img_emb_all: np.ndarray
    if image_paths:
        img_emb_list = get_image_embeddings(image_paths)
        img_emb_all = np.array(img_emb_list, dtype=np.float32)
    else:
        img_emb_all = np.zeros((0, 0), dtype=np.float32)

    # Determine CLIP dimension using a quick probe from text side if needed
    clip_probe = np.array(get_clip_text_embeddings(["probe"]), dtype=np.float32)
    clip_dim = clip_probe.shape[1] if clip_probe.size > 0 else (img_emb_all.shape[1] if img_emb_all.size > 0 else 768)

    # --- Build per-page image vectors (average of images on the page) ---
    page_img_vecs = np.zeros((len(pages_text), clip_dim), dtype=np.float32)
    for p_idx in range(len(pages_text)):
        indices = page_to_img_indices.get(p_idx, [])
        if indices and img_emb_all.size > 0:
            page_img_vecs[p_idx] = np.mean(img_emb_all[indices, :], axis=0)
        else:
            page_img_vecs[p_idx] = np.zeros((clip_dim,), dtype=np.float32)

    # --- Concatenate weighted text + image to get fused page vectors ---
    fused_pages = np.concatenate([
        text_emb * FUSION_TEXT_WEIGHT,
        page_img_vecs * FUSION_IMAGE_WEIGHT,
    ], axis=1)

    index = build_faiss_index(fused_pages)

    # Optional: build image index for fallback retrieval (CLIP space)
    image_index = None
    if img_emb_all.size > 0:
        try:
            image_index = build_faiss_index(img_emb_all.astype(np.float32))
        except Exception:
            image_index = None
    return {
        "pages_text": pages_text,
        "image_paths": image_paths,
        "ocr_snippets": ocr_snippets,
        "text_emb": text_emb,
        "page_img_vecs": page_img_vecs,
        "fused_index": index,
        "clip_dim": clip_dim,
        "img_emb_all": img_emb_all,
        "image_index": image_index,
    }


def build_early_fusion_query_vec(query: str, clip_dim: int) -> np.ndarray:
    q_t = np.array(get_text_embeddings([query]), dtype=np.float32)
    q_c = np.array(get_clip_text_embeddings([query]), dtype=np.float32)
    if q_t.size == 0:
        q_t = np.zeros((1, 768), dtype=np.float32)
    if q_c.size == 0:
        q_c = np.zeros((1, clip_dim), dtype=np.float32)
    q_vec = np.concatenate([
        q_t[0] * FUSION_TEXT_WEIGHT,
        q_c[0] * FUSION_IMAGE_WEIGHT,
    ], axis=0)
    return q_vec.astype(np.float32)


async def answer_one_question_early(query: str, corpus: Dict[str, Any]):
    # One-time build
    if "fused_index" not in corpus:
        raise RuntimeError("Corpus not initialized with fused_index")

    q_vec = build_early_fusion_query_vec(query, corpus["clip_dim"])
    D, I = top_k(corpus["fused_index"], q_vec, min(K_PAGES, len(corpus["pages_text"])))

    # Collect supporting pages and a few images from these pages
    support_texts: List[str] = []
    support_images: List[str] = []

    # Build map page->image paths for quick selection
    page_to_images: DefaultDict[int, List[str]] = defaultdict(list)
    for p in corpus["image_paths"]:
        pi = _page_from_img_path(p)
        if pi is not None:
            page_to_images[pi].append(p)

    for page_idx in I:
        txt = corpus["pages_text"][page_idx][:MAX_TEXT_CHARS]
        support_texts.append(txt)

        # Prefer OCR snippets for this page if available
        page_tag = f"[Page {page_idx + 1}]"
        page_ocr = [s for s in corpus["ocr_snippets"] if page_tag in s]
        support_texts.extend(page_ocr[:1])  # add at most one OCR line

        # Add up to 2 images from this page
        for imgp in page_to_images.get(page_idx, [])[:2]:
            if len(support_images) < 3:  # limit total images to 3
                support_images.append(imgp)

    # Fallback: if we didn't gather any images from retrieved pages, retrieve by CLIP directly
    if len(support_images) < 3 and corpus.get("image_index") is not None and corpus.get("image_paths"):
        try:
            q_clip = np.array(get_clip_text_embeddings([query]), dtype=np.float32)
            if q_clip.size > 0:
                _, I_img = top_k(corpus["image_index"], q_clip[0], min(3, len(corpus["image_paths"])))
                for idx in I_img:
                    pth = corpus["image_paths"][int(idx)]
                    if pth not in support_images and len(support_images) < 3:
                        support_images.append(pth)

                        # Add a short OCR/caption line for this image if available
                        base = os.path.basename(pth).split(".")[0]
                        matched_ocr = next((s for s in corpus["ocr_snippets"] if base in s), None)
                        if matched_ocr:
                            support_texts.append(matched_ocr)
                        else:
                            support_texts.append(f"[Figure match] {base}")
        except Exception:
            pass

    # Optional reranking of textual candidates
    rerank_scores: List[float] = []
    if support_texts:
        rerank_scores = rerank_candidates(query, support_texts)
        order = np.argsort(-np.array(rerank_scores))[:6]
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
    images_to_use = support_images[:3] if corpus.get("include_images", True) else []
    answer = gemini_generate(prompt, image_paths=images_to_use)
    debug = {
        "retrieved_pages": [int(i) + 1 for i in I.tolist()],
        "used_images": support_images[:3],
        "scores": [float(x) for x in D.tolist()],
        "rerank": rerank_scores,
    }
    return answer, debug


# ---------------- Routes ----------------
@app.post("/hackrx/run_earlyfusion", response_model=EFResponse)
async def run_earlyfusion(body: EFRequest, include_images: bool = True):
    input_path = body.documents

    cache_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "_lf_cache")
    os.makedirs(cache_root, exist_ok=True)
    run_dir = os.path.join(cache_root, f"early_{int(time.time()*1000)}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        if _is_url(input_path):
            pdf_path = _download_pdf_to(input_path, cache_root)
        elif os.path.isdir(input_path):
            pdfs = [p for p in os.listdir(input_path) if p.lower().endswith(".pdf")]
            if not pdfs:
                return EFResponse(answers=["[Error: no PDF found in directory]"], debug={"dir": input_path})
            pdf_path = os.path.join(input_path, pdfs[0])
        else:
            if not os.path.exists(input_path):
                return EFResponse(answers=["[Error: document path not found]"], debug={"path": input_path})
            pdf_path = input_path
    except Exception as e:
        return EFResponse(answers=[f"[Error: {e}]"], debug={"path": input_path})

    parsed = extract_text_and_images(pdf_path, run_dir)
    corpus = build_early_fusion_corpus(parsed)
    corpus["include_images"] = include_images

    tasks = [answer_one_question_early(q, corpus) for q in body.questions]
    results = await asyncio.gather(*tasks)
    answers = [r[0] for r in results]
    debug = {
        "per_q": [r[1] for r in results],
        "doc": os.path.basename(pdf_path),
        "pages": len(corpus["pages_text"]),
        "images": len(corpus["image_paths"]),
        "fusion": body.fusion,
        "faiss_gpu": FAISS_USE_GPU and FAISS_HAS_GPU,
        "weights": {"text": FUSION_TEXT_WEIGHT, "image": FUSION_IMAGE_WEIGHT},
        "include_images": include_images,
    }
    return EFResponse(answers=answers, debug=debug)
