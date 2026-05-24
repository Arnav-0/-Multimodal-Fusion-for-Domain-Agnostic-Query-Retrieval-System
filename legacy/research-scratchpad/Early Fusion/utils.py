# utils.py
"""
Client-side utility functions for calling the model server and simple
document utilities (text extraction, image extraction, chunking, and BM25 retrieval).
Designed to be used by the main app (main.py).
"""

import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
import os
import logging

# Document processing
try:
    import fitz  # PyMuPDF
    _has_fitz = True
except Exception:
    _has_fitz = False

try:
    import docx
    _has_docx = True
except Exception:
    _has_docx = False

# For BM25-based context extraction
try:
    from rank_bm25 import BM25Okapi
    _has_bm25 = True
except Exception:
    _has_bm25 = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utils")

# Model server URL (update if different)
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")

# --------------------
# Helpers: PIL <-> base64
def pil_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b

def base64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data)).convert("RGB")

# --------------------
# Model-server clients (synchronous; intended to be called with run_in_executor)
def get_text_embedding_from_server(texts: List[str]) -> np.ndarray:
    """
    Request text embeddings for a list of texts.
    Returns: np.ndarray shape (len(texts), text_dim), dtype float32
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    url = f"{MODEL_SERVER_URL}/get_text_embeddings"
    resp = requests.post(url, json={"texts": texts}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    emb = np.array(data["embeddings"], dtype=np.float32)
    return emb

def get_image_embeddings_from_server(images_data: List[Dict[str, Any]]) -> np.ndarray:
    """
    images_data: list of dicts with key 'image' (PIL.Image) and optional 'page_number'
    Returns: np.ndarray shape (len(images_data), img_dim), dtype float32
    """
    if not images_data:
        return np.zeros((0, 0), dtype=np.float32)

    images_b64 = []
    for item in images_data:
        img = item.get("image")
        if isinstance(img, Image.Image):
            images_b64.append(pil_to_base64(img))
        else:
            # If image is bytes, try to build PIL.Image
            try:
                images_b64.append(pil_to_base64(Image.open(BytesIO(img)).convert("RGB")))
            except Exception:
                raise ValueError("images_data items must contain PIL.Image under 'image' key")

    url = f"{MODEL_SERVER_URL}/get_image_embeddings"
    resp = requests.post(url, json={"images_b64": images_b64}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    emb = np.array(data["embeddings"], dtype=np.float32)
    return emb

def get_visual_explanation_from_server(image: Image.Image, question: str = "") -> str:
    """
    Send a single PIL.Image and question to the server and get back the visual explanation text.
    Returns a string (OCR + caption).
    """
    url = f"{MODEL_SERVER_URL}/get_visual_explanation"
    payload = {"image_b64": pil_to_base64(image), "question": question}
    resp = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    # Combine OCR text + short_caption for a textual explanation
    ocr_text = data.get("ocr_text", "") or ""
    short_caption = data.get("short_caption", "") or ""
    if ocr_text and short_caption:
        return f"{short_caption} — OCR: {ocr_text}"
    elif ocr_text:
        return ocr_text
    elif short_caption:
        return short_caption
    else:
        return ""

# --------------------
# Document helpers
def load_pdf(path: str) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz).
    Returns full text as a single string.
    """
    if not _has_fitz:
        raise ImportError("PyMuPDF (fitz) is required for load_pdf. Install with `pip install pymupdf`.")
    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)

def extract_images_from_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Extract images from PDF using PyMuPDF.
    Returns list of dicts: {"image": PIL.Image, "page_number": int}
    If no images are embedded, this function will fallback to rendering pages to images (one image per page).
    """
    images = []
    if not _has_fitz:
        raise ImportError("PyMuPDF (fitz) is required for extract_images_from_pdf. Install with `pip install pymupdf`.")
    doc = fitz.open(path)
    for pageno in range(len(doc)):
        page = doc[pageno]
        # Extract embedded images
        image_list = page.get_images(full=True)
        if image_list:
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                images.append({"image": pil_img, "page_number": pageno + 1})
        else:
            # No embedded images — fallback to rendering the page as an image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # higher res
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            images.append({"image": pil_img, "page_number": pageno + 1})
    doc.close()
    return images

def load_docx(path: str) -> str:
    """
    Extract text from a DOCX file using python-docx
    """
    if not _has_docx:
        raise ImportError("python-docx is required for load_docx. Install with `pip install python-docx`.")
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

# --------------------
# Chunking and retrieval helpers
def chunk_text(text: str, chunk_size: int = 600, overlap: int = 50) -> List[str]:
    """
    Chunk text into overlapping chunks by word count.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks

def extract_relevant_context(full_text: str, question: str, max_length: int = 2000) -> str:
    """
    Simple BM25 sentence selection: split document into sentences/paragraphs,
    rank them by BM25 score against the question, and return the concatenated
    top passages up to max_length characters.
    Requires rank_bm25.
    """
    if not _has_bm25:
        # fallback: return the first max_length chars
        return full_text[:max_length]

    # split into paragraphs (or sentences)
    passages = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    if not passages:
        passages = [s.strip() for s in full_text.split(". ") if s.strip()]

    tokenized = [p.split() for p in passages]
    bm25 = BM25Okapi(tokenized)
    q_tokens = question.split()
    scores = bm25.get_scores(q_tokens)
    ranked_idx = np.argsort(scores)[::-1].tolist()

    selected = []
    cur_len = 0
    for idx in ranked_idx:
        p = passages[idx]
        if cur_len + len(p) > max_length:
            continue
        selected.append(p)
        cur_len += len(p)
        if cur_len >= max_length:
            break
    if not selected:
        return full_text[:max_length]
    return "\n\n".join(selected)

# --------------------
# If module executed directly, run a quick local test (optional)
if __name__ == "__main__":
    print("Utils module loaded. MODEL_SERVER_URL =", MODEL_SERVER_URL)
    # Minimal sanity checks could go here, but avoid running heavy tests automatically.
