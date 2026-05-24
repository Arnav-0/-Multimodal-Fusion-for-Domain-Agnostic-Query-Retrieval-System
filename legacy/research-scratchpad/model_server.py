# File: model_server.py
import io
import base64
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from typing import List, Dict

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Models ---
class TextEmbeddingRequest(BaseModel):
    texts: List[str]

class VQARequest(BaseModel):
    image_b64: str
    question: str
    
class RerankRequest(BaseModel):
    query: str
    documents: List[str]

# --- Model Loading ---
try:
    logger.info("Loading CLIP model (for retrieval)...")
    retrieval_model = SentenceTransformer('clip-ViT-B-32')
    logger.info("✅ CLIP model loaded successfully.")

    logger.info("Loading CrossEncoder model (for re-ranking)...")
    rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("✅ CrossEncoder model loaded successfully.")

    logger.info("Loading Vision Question Answering (VQA) model (LLaVA)...")
    vqa_pipeline = pipeline(
        "image-to-text",
        model="llava-hf/llava-1.5-7b-hf",
        model_kwargs={"device_map": "auto", "load_in_4bit": True}
    )
    logger.info("✅ VQA model loaded successfully.")
except Exception as e:
    logger.error(f"Fatal error during model loading: {e}", exc_info=True)
    retrieval_model, rerank_model, vqa_pipeline = None, None, None

app = FastAPI(title="AI Model Inference Server")

@app.on_event("startup")
async def startup_event():
    if not all([retrieval_model, rerank_model, vqa_pipeline]):
        raise RuntimeError("One or more models did not load correctly. Please check logs.")
    logger.info("✅ Model Server is ready and all models are in memory.")

# --- Helper Function ---
def get_visual_explanation(image: Image.Image, question: str) -> str:
    prompt = (
        "USER: <image>\n"
        "You are an expert technical analyst. Your task is to meticulously analyze the provided diagram.\n"
        "First, describe all the labeled components and connections you see in the diagram step-by-step. "
        "Then, based only on your detailed description, provide a final answer to the user's question.\n"
        f"Question: {question}\n"
        "ASSISTANT:"
    )
    try:
        explanation = vqa_pipeline(image, prompt=prompt, generate_kwargs={"max_new_tokens": 250})
        return explanation[0]['generated_text'].split("ASSISTANT:")[1].strip()
    except Exception as e:
        logger.error(f"Error during visual explanation: {e}")
        return "An error occurred while analyzing the image."

# --- API Endpoints ---
@app.post("/get_text_embedding")
def get_text_embedding(request: TextEmbeddingRequest):
    try:
        embeddings = retrieval_model.encode(request.texts).tolist()
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rerank")
def rerank(request: RerankRequest):
    try:
        # The cross-encoder expects pairs of [query, document]
        model_input = [[request.query, doc] for doc in request.documents]
        scores = rerank_model.predict(model_input)
        
        # Combine documents with their scores and sort
        scored_docs = list(zip(request.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the text of the re-ranked documents
        reranked_docs = [doc for doc, score in scored_docs]
        return {"reranked_documents": reranked_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_visual_explanation")
def get_explanation(request: VQARequest):
    try:
        image_bytes = base64.b64decode(request.image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        explanation = get_visual_explanation(image, request.question)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)