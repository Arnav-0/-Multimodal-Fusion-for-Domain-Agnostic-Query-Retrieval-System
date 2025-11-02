import os
import logging
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------- Logging ----------------
app = FastAPI(title="Model Server (Embeddings + Rerank)")
logger = logging.getLogger("model_server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------- Device / Config ----------------
DEVICE = os.getenv("MODEL_DEVICE", "cuda").lower()
DEVICE_ID = int(os.getenv("MODEL_DEVICE_ID", "0"))

if DEVICE == "cuda":
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        DEVICE = "cpu"
    else:
        torch.cuda.set_device(DEVICE_ID)
        gpu_name = torch.cuda.get_device_name(DEVICE_ID)
        gpu_memory = torch.cuda.get_device_properties(DEVICE_ID).total_memory / 1024**3
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB) on device {DEVICE_ID}")
        
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy for better GPU memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
else:
    logger.info(f"Using CPU for inference")

TEXT_EMB_MODEL_NAME = os.getenv("TEXT_EMB_MODEL_NAME", "intfloat/e5-base")
CLIP_MODEL_NAME     = os.getenv("CLIP_MODEL_NAME", "clip-ViT-L-14")  # better quality than B-32
CROSS_ENCODER_NAME  = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- Load models ----------------
logger.info(f"Loading text model: {TEXT_EMB_MODEL_NAME} on {DEVICE}")
text_model = SentenceTransformer(TEXT_EMB_MODEL_NAME, device=DEVICE)

logger.info(f"Loading CLIP model: sentence-transformers/{CLIP_MODEL_NAME} on {DEVICE}")
clip_model = SentenceTransformer(f"sentence-transformers/{CLIP_MODEL_NAME}", device=DEVICE)

logger.info(f"Loading cross-encoder: {CROSS_ENCODER_NAME} on {DEVICE}")
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device=DEVICE)

# ---------------- Schemas ----------------
class Texts(BaseModel):
    texts: List[str]

class ImagePaths(BaseModel):
    paths: List[str]

class RerankBody(BaseModel):
    query: str
    candidates: List[str]

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    """Health check with GPU info"""
    response = {
        "status": "ok",
        "device": DEVICE,
        "device_id": DEVICE_ID if DEVICE == "cuda" else None
    }
    
    if DEVICE == "cuda" and torch.cuda.is_available():
        response.update({
            "gpu_name": torch.cuda.get_device_name(DEVICE_ID),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(DEVICE_ID).total_memory / 1024**3, 2),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(DEVICE_ID) / 1024**3, 2),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(DEVICE_ID) / 1024**3, 2)
        })
    
    return response

@app.post("/get_text_embeddings")
def api_get_text_embeddings(body: Texts):
    emb = text_model.encode(body.texts, convert_to_numpy=True, normalize_embeddings=True)
    return {"embeddings": emb.astype(np.float32).tolist()}

@app.post("/get_clip_text_embeddings")
def api_get_clip_text_embeddings(body: Texts):
    emb = clip_model.encode(body.texts, convert_to_numpy=True, normalize_embeddings=True)
    return {"embeddings": emb.astype(np.float32).tolist()}

@app.post("/get_image_embeddings")
def api_get_image_embeddings(body: ImagePaths):
    images = []
    for p in body.paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (255, 255, 255))  # keep dims stable
        images.append(img)
    emb = clip_model.encode(images, convert_to_numpy=True, normalize_embeddings=True)
    return {"embeddings": emb.astype(np.float32).tolist()}

@app.post("/rerank")
def api_rerank(body: RerankBody):
    pairs = [[body.query, c] for c in body.candidates]
    scores = cross_encoder.predict(pairs).astype(float).tolist()
    return {"scores": scores}
