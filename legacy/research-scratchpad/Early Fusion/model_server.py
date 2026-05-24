# model_server.py
"""
FastAPI model/inference server for multimodal embeddings + visual explanation.

Endpoints:
- POST /get_text_embeddings  -> returns embeddings for a list of texts
- POST /get_image_embeddings -> returns embeddings for a list of base64 images
- POST /get_visual_explanation -> returns OCR+basic analysis for an image + question
- GET  /health -> simple health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import logging

# SentTrans (CLIP) for joint image/text space
from sentence_transformers import SentenceTransformer

# Try to import OCR backends (optional)
try:
    import easyocr
    _has_easyocr = True
except Exception:
    _has_easyocr = False

try:
    import pytesseract
    _has_pytesseract = True
except Exception:
    _has_pytesseract = False

# --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_server")

app = FastAPI(title="Multimodal Model Server")

# --------------------
# Pydantic request/response models
class TextsRequest(BaseModel):
    texts: List[str]

class ImagesRequest(BaseModel):
    images_b64: List[str]

class VisualExplainRequest(BaseModel):
    image_b64: str
    question: str = ""

class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]

class VisualExplainResponse(BaseModel):
    ocr_text: str
    short_caption: str
    metadata: Dict[str, Any]

# --------------------
# Load retrieval model (CLIP-style) once
MODEL_NAME = os.getenv("RETRIEVAL_MODEL_NAME", "clip-ViT-B-32")
DEVICE = os.getenv("MODEL_DEVICE", "cpu")  # set "cuda" if available

logger.info(f"Loading retrieval model: {MODEL_NAME} on {DEVICE} ...")
retrieval_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
logger.info("Retrieval model loaded.")

# If easyocr is available, create reader
if _has_easyocr:
    logger.info("EasyOCR available — initializing reader.")
    try:
        _ocr_reader = easyocr.Reader(["en"], gpu=(DEVICE.startswith("cuda")))
    except Exception as e:
        logger.warning(f"EasyOCR initialization failed: {e}")
        _ocr_reader = None
else:
    _ocr_reader = None

# --------------------
# Helpers
def decode_b64_to_pil(b64str: str) -> Image.Image:
    try:
        data = base64.b64decode(b64str)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")

def pil_to_numpy_list(img: Image.Image) -> List[float]:
    """Return image bytes as raw list (not used for embedding but kept for debugging)."""
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return list(buf.getvalue())

# --------------------
# Endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "easyocr": bool(_ocr_reader), "pytesseract": _has_pytesseract}

@app.post("/get_text_embeddings", response_model=EmbeddingsResponse)
def get_text_embeddings(req: TextsRequest):
    """
    Return embeddings for a list of texts using the retrieval_model.encode
    """
    try:
        if not req.texts:
            return {"embeddings": []}
        # SentenceTransformer supports batching internally
        emb = retrieval_model.encode(req.texts, convert_to_numpy=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        return {"embeddings": emb.tolist()}
    except Exception as e:
        logger.exception("Text embedding failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_image_embeddings", response_model=EmbeddingsResponse)
def get_image_embeddings(req: ImagesRequest):
    """
    Accept list of base64-encoded images, decode, and return embeddings.
    """
    try:
        if not req.images_b64:
            return {"embeddings": []}
        pil_imgs = []
        for b64 in req.images_b64:
            pil_imgs.append(decode_b64_to_pil(b64))

        emb = retrieval_model.encode(pil_imgs, convert_to_numpy=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        return {"embeddings": emb.tolist()}
    except Exception as e:
        logger.exception("Image embedding failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_visual_explanation", response_model=VisualExplainResponse)
def get_visual_explanation(req: VisualExplainRequest):
    """
    Very lightweight visual explanation:
    - Run OCR (EasyOCR if available, else pytesseract if available)
    - Return OCR text and a short placeholder caption (we keep this small and deterministic)
    This endpoint is intended for compatibility with pipelines that previously asked for visual explanations.
    """
    try:
        img = decode_b64_to_pil(req.image_b64)

        # OCR: prefer EasyOCR
        ocr_text = ""
        if _ocr_reader is not None:
            try:
                ocr_res = _ocr_reader.readtext(np.array(img))
                # easyocr returns list of (bbox, text, confidence)
                ocr_text = " ".join([t[1] for t in ocr_res if len(t) >= 2 and t[1]])
            except Exception as e:
                logger.warning(f"EasyOCR read failed: {e}")
                ocr_text = ""
        elif _has_pytesseract:
            try:
                import pytesseract
                ocr_text = pytesseract.image_to_string(img)
            except Exception as e:
                logger.warning(f"pytesseract OCR failed: {e}")
                ocr_text = ""
        else:
            ocr_text = ""

        # Basic captioning heuristic (very small): approximate by returning top-k colors and size
        w, h = img.size
        # compute average color
        arr = np.array(img).reshape(-1, 3).astype(np.float32)
        avg_col = np.mean(arr, axis=0).astype(int).tolist()
        short_caption = f"Image {w}x{h}, avg_color={avg_col}"

        metadata = {"width": w, "height": h, "ocr_length": len(ocr_text)}
        return {"ocr_text": ocr_text, "short_caption": short_caption, "metadata": metadata}
    except Exception as e:
        logger.exception("Visual explanation failed")
        raise HTTPException(status_code=500, detail=str(e))
