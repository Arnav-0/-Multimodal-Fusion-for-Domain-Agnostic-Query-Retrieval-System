# main.py -- Early Fusion (feature-level concatenation) implementation
"""
Advanced Multimodal Document Q&A System - Early Fusion (Feature-level)

This version uses a single embedding backend for both text and images,
concatenates text and image embeddings (feature-level early fusion), and
indexes the fused vectors in FAISS for retrieval.
"""

import asyncio
import logging
import os
import ssl
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import certifi
import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Local utils (must provide these functions)
from utils import (
    extract_images_from_pdf,
    get_image_embeddings_from_server,
    get_text_embedding_from_server,
    get_visual_explanation_from_server,  # kept for optional debug / future use
    load_docx,
    load_pdf,
    chunk_text,
    extract_relevant_context,
)

# --- INITIAL SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

# --- CONFIG / GLOBALS ---
REQUEST_TIMEOUT = 300
API_KEYS = [key for key in [os.getenv(f"COHERE_API_KEY_{i}") for i in range(1, 32)] if key]
if not API_KEYS:
    # generation still needs a LM key; if you use a different LLM provider change accordingly
    raise ValueError("No COHERE_API_KEY found in environment variables.")

app = FastAPI(title="Early-Fusion Multimodal Q&A")
executor = ThreadPoolExecutor(max_workers=8)
document_cache: Dict[str, Dict[str, Any]] = {}
current_key_index = 0

# --- MODELS / API MODELS ---
class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# --- Cohere generation wrapper (unchanged) ---
async def cohere_api_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    global current_key_index
    for _ in range(len(API_KEYS)):
        try:
            api_key = API_KEYS[current_key_index]
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as session:
                async with session.post(f"https://api.cohere.ai/v1/{endpoint}", headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    resp.raise_for_status()
        except Exception as e:
            logger.warning(f"API key #{current_key_index + 1} failed: {e}. Switching key.")
            current_key_index = (current_key_index + 1) % len(API_KEYS)
    raise HTTPException(status_code=429, detail="All Cohere API keys failed.")

# --- Utility: Save markdown session (keeps previous behavior) ---
def save_session_to_markdown(doc_url: str, qa_pairs: List[Dict[str, Any]]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qa_session_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Q&A Session Report\n\n**Document URL:** {doc_url}\n\n---\n\n")
        for item in qa_pairs:
            f.write(f"## Question {item['id']}\n**Question:** {item['question']}\n\n**Answer:**\n{item['answer']}\n\n")
            f.write(f"### Context Used:\n#### Text Context\n```\n{item['text_context']}\n```\n")
            if item.get("image_context"):
                f.write(f"#### Visual Context\n```\n{item['image_context']}\n```\n")
            f.write("---\n\n")
    logger.info(f"💾 Q&A session saved to {filename}")

# --- Main processing endpoint (Early Fusion) ---
@app.post("/hackrx/run", response_model=QAResponse)
async def qa_from_document(body: QARequest):
    start_time = time.time()
    loop = asyncio.get_event_loop()
    doc_input = body.documents

    # If cached, reuse
    if doc_input in document_cache:
        logger.info("🚀 Cache hit! Using cached fused index.")
        cached_data = document_cache[doc_input]
    else:
        logger.info("Cache miss. Processing document for early fusion.")

        # Save/download document
        if doc_input.startswith(("http://", "https://")):
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(doc_input) as response:
                    response.raise_for_status()
                    content = await response.read()
            save_dir = Path("saved_docs")
            save_dir.mkdir(exist_ok=True)
            file_ext = Path(doc_input.split("?")[0]).suffix or ".pdf"
            save_path = save_dir / f"{uuid.uuid4().hex}{file_ext}"
            save_path.write_bytes(content)
        else:
            save_path = Path(doc_input)
            if not save_path.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {doc_input}")
            file_ext = save_path.suffix.lower()
            logger.info(f"📄 Using local file: {save_path}")

        # Extract text
        logger.info("🔍 Extracting text from document...")
        text = await loop.run_in_executor(executor, load_pdf if file_ext == ".pdf" else load_docx, str(save_path))
        full_text = text.strip()
        if not full_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        logger.info(f"📄 Extracted text length: {len(full_text)} characters")

        # Chunk text (granularity for retrieval)
        chunks = chunk_text(full_text, chunk_size=600, overlap=50)
        logger.info(f"📚 Text split into {len(chunks)} chunks")

        # 1) Text embeddings (model server)
        try:
            text_embeddings = await loop.run_in_executor(executor, get_text_embedding_from_server, chunks)
            text_embeddings_np = np.array(text_embeddings).astype("float32")  # (num_chunks, text_dim)
            text_dim = text_embeddings_np.shape[1]
            logger.info(f"🔢 Text embeddings shape: {text_embeddings_np.shape}")
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise HTTPException(status_code=500, detail="Text embedding failed")

        # 2) Extract images from document and compute image embeddings (doc-time)
        logger.info("🖼️ Extracting images from document...")
        images_data = await loop.run_in_executor(executor, extract_images_from_pdf, str(save_path))
        num_images = len(images_data) if images_data else 0
        logger.info(f"🖼️ Found {num_images} images")

        image_embeddings_np = None
        if num_images > 0:
            try:
                image_embeddings = await loop.run_in_executor(executor, get_image_embeddings_from_server, images_data)
                image_embeddings_np = np.array(image_embeddings).astype("float32")  # (num_images, img_dim)
                img_dim = image_embeddings_np.shape[1]
                logger.info(f"🔢 Image embeddings shape: {image_embeddings_np.shape}")
            except Exception as e:
                logger.warning(f"Image embedding failed: {e}. Proceeding with no image part.")
                image_embeddings_np = None
                img_dim = 0
        else:
            image_embeddings_np = None
            img_dim = 0

        # 3) Pool image embeddings to a single document-level image vector (mean pooling)
        if image_embeddings_np is not None and image_embeddings_np.shape[0] > 0:
            pooled_image_emb = np.mean(image_embeddings_np, axis=0).astype("float32")  # (img_dim,)
            logger.info(f"🔗 Pooled image embedding dim: {pooled_image_emb.shape[0]}")
        else:
            pooled_image_emb = None
            logger.info("🔗 No image embeddings to pool (will zero-pad image part)")

        # 4) Build fused vectors (concat text_emb + pooled_image_emb or zeros)
        fused_vectors_list = []
        fused_metadata = []
        if img_dim == 0:
            # No image part: fused vectors are just text embeddings
            fused_vectors_np = text_embeddings_np.copy()
            for i, chunk in enumerate(chunks):
                fused_metadata.append({"chunk": chunk, "chunk_id": i, "has_image": False})
            fused_dim = fused_vectors_np.shape[1]
        else:
            # Concatenate per chunk: [text_vec | pooled_image_emb]
            zero_img = np.zeros((img_dim,), dtype="float32")
            for i, txt_vec in enumerate(text_embeddings_np):
                img_part = pooled_image_emb if pooled_image_emb is not None else zero_img
                fused = np.concatenate([txt_vec, img_part], axis=0).astype("float32")
                fused_vectors_list.append(fused)
                fused_metadata.append({"chunk": chunks[i], "chunk_id": i, "has_image": True})
            fused_vectors_np = np.vstack(fused_vectors_list)
            fused_dim = fused_vectors_np.shape[1]

        logger.info(f"🧩 Fused vectors shape: {fused_vectors_np.shape} (dim={fused_dim})")

        # 5) Normalize and build FAISS index (cosine similarity via normalized inner-product)
        faiss.normalize_L2(fused_vectors_np)
        fused_index = faiss.IndexFlatIP(fused_dim)
        fused_index.add(fused_vectors_np)
        logger.info(f"🗃️ Fused FAISS index created with {fused_index.ntotal} vectors (dim={fused_dim})")

        # 6) Cache
        cached_data = {
            "fused_index": fused_index,
            "fused_vectors": fused_vectors_np,
            "fused_metadata": fused_metadata,
            "full_text": full_text,
            "images_data": images_data,
            "text_dim": text_dim,
            "img_dim": img_dim,
            "chunks": chunks,
        }
        document_cache[doc_input] = cached_data

    # --- Process questions using fused retrieval ---
    answers = []
    qa_results_for_saving = []

    for i, question in enumerate(body.questions):
        logger.info(f"Processing question {i+1}/{len(body.questions)}: {question}")

        # Embed question with same text encoder
        try:
            q_text_emb = await loop.run_in_executor(executor, get_text_embedding_from_server, [question])
            q_text_emb = np.array(q_text_emb).astype("float32")[0]  # (text_dim,)
        except Exception as e:
            logger.error(f"Question embedding failed: {e}")
            raise HTTPException(status_code=500, detail="Question embedding failed")

        # Create fused query: concat(q_text_emb, zeros_for_image_part)
        if cached_data["img_dim"] == 0:
            q_fused = q_text_emb
        else:
            q_img_part = np.zeros((cached_data["img_dim"],), dtype="float32")
            q_fused = np.concatenate([q_text_emb, q_img_part], axis=0).astype("float32")

        # Normalize and search
        faiss.normalize_L2(q_fused.reshape(1, -1))
        D, I = cached_data["fused_index"].search(q_fused.reshape(1, -1), k=5)
        top_idx = I[0].tolist()
        top_chunks = [cached_data["fused_metadata"][idx]["chunk"] for idx in top_idx]

        # Build context for LLM: join top chunks; include doc-level pooled image info as a short descriptor if available
        relevant_text = "\n\n".join(top_chunks)
        image_context_descr = ""
        if cached_data["img_dim"] > 0 and cached_data.get("images_data"):
            image_context_descr = "Document contains charts/tables/figures (pre-encoded as image embeddings)."

        final_prompt = f"""You are an expert document analyst. Use the provided text context and the pre-encoded visual context to answer the question.

DOCUMENT TEXT CONTEXT:
{relevant_text}

PRE-ENCODED VISUAL CONTEXT:
{image_context_descr if image_context_descr else 'No visual elements detected.'}

QUESTION:
{question}

Provide a concise, accurate answer, mention any numbers explicitly, and say if information is missing.
"""

        # Generate answer with Cohere (same as before)
        answer_data = await cohere_api_request("generate", {
            "prompt": final_prompt,
            "model": "command-r-plus",
            "max_tokens": 400,
            "temperature": 0.1,
            "truncate": "END"
        })
        answer = answer_data["generations"][0]["text"].strip()

        answers.append(answer)
        qa_results_for_saving.append({
            "id": i + 1,
            "question": question,
            "answer": answer,
            "text_context": relevant_text,
            "image_context": image_context_descr,
        })
        logger.info(f"Answer {i+1} ready.")

    # Save session and return
    save_session_to_markdown(doc_input, qa_results_for_saving)
    total_time = time.time() - start_time
    logger.info(f"🎉 All questions processed in {total_time:.2f} s.")
    return {"answers": answers}

# --- Streaming endpoint (mirrors above but streaming) ---
@app.post("/hackrx/stream")
async def qa_stream_from_document(body: QARequest):
    async def generate_stream():
        start_time = time.time()
        loop = asyncio.get_event_loop()
        doc_input = body.documents

        yield f"data: {{'status': 'processing_document', 'message': 'Starting...'}}\n\n"

        if doc_input in document_cache:
            cached_data = document_cache[doc_input]
            yield f"data: {{'status': 'cache_hit', 'message': 'Using cached fused index'}}\n\n"
        else:
            yield f"data: {{'status': 'processing', 'message': 'Building fused index...'}}\n\n"
            # (Reuse same processing logic from /hackrx/run but simplified streaming)
            try:
                # load/save doc
                if doc_input.startswith(("http://", "https://")):
                    ssl_context = ssl.create_default_context(cafile=certifi.where())
                    connector = aiohttp.TCPConnector(ssl=ssl_context)
                    async with aiohttp.ClientSession(connector=connector) as session:
                        async with session.get(doc_input) as response:
                            response.raise_for_status()
                            content = await response.read()
                    save_dir = Path("saved_docs")
                    save_dir.mkdir(exist_ok=True)
                    file_ext = Path(doc_input.split("?")[0]).suffix or ".pdf"
                    save_path = save_dir / f"{uuid.uuid4().hex}{file_ext}"
                    save_path.write_bytes(content)
                else:
                    save_path = Path(doc_input)
                    if not save_path.exists():
                        yield f"data: {{'status': 'error', 'message': 'File not found'}}\n\n"
                        return
                    file_ext = save_path.suffix.lower()

                # extract text
                yield f"data: {{'status': 'extracting_text', 'message': 'Extracting text...'}}\n\n"
                text = await loop.run_in_executor(executor, load_pdf if file_ext == ".pdf" else load_docx, str(save_path))
                full_text = text.strip()
                if not full_text:
                    yield f"data: {{'status': 'error', 'message': 'No text extracted'}}\n\n"
                    return

                # chunk text
                chunks = chunk_text(full_text, chunk_size=600, overlap=50)
                yield f"data: {{'status': 'chunked', 'n_chunks': {len(chunks)}}}\n\n"

                # text embeddings
                yield f"data: {{'status': 'text_embedding', 'message': 'Embedding text...'}}\n\n"
                text_embeddings = await loop.run_in_executor(executor, get_text_embedding_from_server, chunks)
                text_embeddings_np = np.array(text_embeddings).astype("float32")

                # images
                yield f"data: {{'status': 'extract_images', 'message': 'Extracting images...'}}\n\n"
                images_data = await loop.run_in_executor(executor, extract_images_from_pdf, str(save_path))
                if images_data:
                    image_embeddings = await loop.run_in_executor(executor, get_image_embeddings_from_server, images_data)
                    image_embeddings_np = np.array(image_embeddings).astype("float32")
                    pooled_image_emb = np.mean(image_embeddings_np, axis=0).astype("float32")
                    img_dim = pooled_image_emb.shape[0]
                else:
                    pooled_image_emb = None
                    img_dim = 0

                # fused vectors
                yield f"data: {{'status': 'fusing', 'message': 'Creating fused vectors...'}}\n\n"
                if img_dim == 0:
                    fused_vectors_np = text_embeddings_np.copy()
                    fused_metadata = [{"chunk": chunks[i], "chunk_id": i, "has_image": False} for i in range(len(chunks))]
                else:
                    fused_vectors_list = []
                    fused_metadata = []
                    for i, tvec in enumerate(text_embeddings_np):
                        fused = np.concatenate([tvec, pooled_image_emb], axis=0).astype("float32")
                        fused_vectors_list.append(fused)
                        fused_metadata.append({"chunk": chunks[i], "chunk_id": i, "has_image": True})
                    fused_vectors_np = np.vstack(fused_vectors_list)

                faiss.normalize_L2(fused_vectors_np)
                fused_index = faiss.IndexFlatIP(fused_vectors_np.shape[1])
                fused_index.add(fused_vectors_np)

                cached_data = {
                    "fused_index": fused_index,
                    "fused_vectors": fused_vectors_np,
                    "fused_metadata": fused_metadata,
                    "full_text": full_text,
                    "images_data": images_data,
                    "text_dim": text_embeddings_np.shape[1],
                    "img_dim": img_dim,
                    "chunks": chunks,
                }
                document_cache[doc_input] = cached_data
                yield f"data: {{'status': 'indexed', 'message': 'Fused index created'}}\n\n"
            except Exception as e:
                yield f"data: {{'status': 'error', 'message': 'Processing failed: {str(e)}'}}\n\n"
                return

        # For each question, do fused retrieval + generate
        for i, question in enumerate(body.questions):
            yield f"data: {{'status': 'processing_question', 'question_num': {i+1}}}\n\n"
            try:
                q_text_emb = await loop.run_in_executor(executor, get_text_embedding_from_server, [question])
                q_text_emb = np.array(q_text_emb).astype("float32")[0]
            except Exception as e:
                yield f"data: {{'status': 'error', 'message': 'Question embedding failed'}}\n\n"
                continue

            if cached_data["img_dim"] == 0:
                q_fused = q_text_emb
            else:
                q_img_part = np.zeros((cached_data["img_dim"],), dtype="float32")
                q_fused = np.concatenate([q_text_emb, q_img_part], axis=0).astype("float32")

            faiss.normalize_L2(q_fused.reshape(1, -1))
            D, I = cached_data["fused_index"].search(q_fused.reshape(1, -1), k=5)
            top_idx = I[0].tolist()
            top_chunks = [cached_data["fused_metadata"][idx]["chunk"] for idx in top_idx]
            relevant_text = "\n\n".join(top_chunks)
            image_context_descr = ""
            if cached_data["img_dim"] > 0 and cached_data.get("images_data"):
                image_context_descr = "Document contains pre-encoded visual elements."

            final_prompt = f"""Answer based on the following context:

CONTEXT:
{relevant_text}

VISUAL: {image_context_descr if image_context_descr else 'No visuals'}

QUESTION: {question}

Give a concise answer with numbers if present.
"""
            # generate
            answer_data = await cohere_api_request("generate", {
                "prompt": final_prompt,
                "model": "command-r-plus",
                "max_tokens": 250,
                "temperature": 0.1
            })
            answer = answer_data["generations"][0]["text"].strip()
            yield f"data: {{'status': 'answer_ready', 'question_num': {i+1}, 'answer': '{answer.replace(chr(10), chr(32)).replace(chr(13), chr(32))}'}}\n\n"

        total_time = time.time() - start_time
        yield f"data: {{'status': 'complete', 'message': 'All questions processed', 'total_time': {total_time:.2f}}}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/plain")

# --- Clear cache endpoint ---
@app.post("/clear_cache")
def clear_cache():
    document_cache.clear()
    logger.info("Cache cleared.")
    return {"status": "cache cleared"}

# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Early-Fusion Web App Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
