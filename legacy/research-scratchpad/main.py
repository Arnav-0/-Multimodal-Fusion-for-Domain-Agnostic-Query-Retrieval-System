# File: main.py
import os
import time
import uuid
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import ssl
import certifi
from datetime import datetime

import cohere
import faiss
import numpy as np
import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi # CORRECTED IMPORT

from utils import (
    load_pdf, load_docx, chunk_text, extract_images_from_pdf,
    get_image_embeddings_from_server, get_visual_explanation_from_server,
    get_text_embedding_from_server
)

# --- INITIAL SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- CONFIGURATION & GLOBAL OBJECTS ---
REQUEST_TIMEOUT = 300
API_KEYS = [key for key in [os.getenv(f"COHERE_API_KEY_{i}") for i in range(1, 32)] if key]
if not API_KEYS:
    raise ValueError("No COHERE_API_KEY found in environment variables.")

app = FastAPI(title="Advanced Multimodal Document Q&A System")
executor = ThreadPoolExecutor(max_workers=8)
current_key_index = 0
document_cache = {}

# --- API MODELS ---
class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# --- ASYNC COHERE WRAPPER ---
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

# --- HELPER FUNCTIONS ---
def save_session_to_markdown(doc_url: str, qa_pairs: List[Dict[str, Any]]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qa_session_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Q&A Session Report\n\n**Document URL:** {doc_url}\n\n---\n\n")
        for item in qa_pairs:
            f.write(f"## Question {item['id']}\n**Question:** {item['question']}\n\n**Answer:**\n{item['answer']}\n\n")
            f.write(f"### Context Used:\n#### Text Context\n```\n{item['text_context']}\n```\n")
            if item['image_context']:
                f.write(f"#### Visual Context\n```\n{item['image_context']}\n```\n")
            f.write("---\n\n")
    logger.info(f"💾 Q&A session saved to {filename}")

# --- MAIN APP LOGIC ---
@app.post("/hackrx/run", response_model=QAResponse)
async def qa_from_document(body: QARequest):
    start_time = time.time()
    loop = asyncio.get_event_loop()
    doc_url = body.documents

    if doc_url in document_cache:
        logger.info("🚀 Cache hit! Using cached indexes and data.")
        cached_data = document_cache[doc_url]
    else:
        logger.info(" Cache miss. Processing document from scratch.")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(doc_url) as response:
                response.raise_for_status()
                content = await response.read()

        save_dir = Path("saved_docs")
        save_dir.mkdir(exist_ok=True)
        file_ext = Path(doc_url.split('?')[0]).suffix or '.pdf'
        save_path = save_dir / f"{uuid.uuid4().hex}{file_ext}"
        save_path.write_bytes(content)

        text = await loop.run_in_executor(executor, load_pdf if file_ext == '.pdf' else load_docx, str(save_path))
        chunks = await loop.run_in_executor(executor, chunk_text, text)
        
        tokenized_chunks = [doc.split(" ") for doc in chunks]
        print(tokenized_chunks)  # DEBUGGING LINE
        bm25_index = BM25Okapi(tokenized_chunks) # CORRECTED INSTANTIATION

        text_embed_payload = {"texts": chunks, "model": "embed-english-v3.0", "input_type": "search_document"}
        text_embeddings = (await cohere_api_request("embed", text_embed_payload))["embeddings"]
        text_embeddings_np = np.array(text_embeddings).astype('float32')
        text_index = faiss.IndexFlatIP(text_embeddings_np.shape[1])
        faiss.normalize_L2(text_embeddings_np)
        text_index.add(text_embeddings_np)

        images_data = await loop.run_in_executor(executor, extract_images_from_pdf, str(save_path))
        image_embeddings_np = await loop.run_in_executor(executor, get_image_embeddings_from_server, images_data)
        
        image_index = None
        if image_embeddings_np is not None and len(image_embeddings_np) > 0:
            image_index = faiss.IndexFlatIP(image_embeddings_np.shape[1])
            faiss.normalize_L2(image_embeddings_np)
            image_index.add(image_embeddings_np)

        cached_data = {
            "text_index": text_index, "chunks": chunks, "bm25_index": bm25_index,
            "image_index": image_index, "images_data": images_data
        }
        document_cache[doc_url] = cached_data

    # --- Process Questions ---
    answers, qa_results_for_saving = [], []
    for i, question in enumerate(body.questions):
        logger.info(f"Processing question {i+1}/{len(body.questions)}: '{question}'")

        # A) Hybrid Search for Text
        bm25_scores = cached_data["bm25_index"].get_scores(question.split(" "))
        bm25_indices = np.argsort(bm25_scores)[::-1][:10]
        
        query_embed = (await cohere_api_request("embed", {"texts": [question], "model": "embed-english-v3.0", "input_type": "search_query"}))["embeddings"]
        _, faiss_indices = cached_data["text_index"].search(np.array(query_embed).astype('float32'), 10)
        
        rrf_scores, k = {}, 60
        for rank, doc_id in enumerate(list(bm25_indices) + list(faiss_indices[0])):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        fused_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:3]
        text_context = "\n\n".join([cached_data["chunks"][idx] for idx in fused_indices])

        # B) Visual Context Retrieval and Explanation
        image_context = ""
        if cached_data["image_index"] is not None:
            query_image_embed_np = await loop.run_in_executor(executor, get_text_embedding_from_server, [question])
            if query_image_embed_np is not None:
                faiss.normalize_L2(query_image_embed_np)
                _, I_img = cached_data["image_index"].search(query_image_embed_np, 1)
                image_data = cached_data["images_data"][I_img[0][0]]
                explanation = await loop.run_in_executor(executor, get_visual_explanation_from_server, image_data["image"], question)
                image_context = f"[Visual Context from Page {image_data['page_number']}]:\n{explanation}"

        # C) Fusion and Final Answer Generation
        final_prompt = (
        "You are an expert analytics assistant specialized in interpreting dashboards, charts, and structured reports. "
        "Your task is to answer the user's question concisely and precisely using the given text and/or images. "
        "Always ground your answers in the provided context. "
        "If the answer is not available in the context, respond with 'The information is not available in the provided data.'\n\n"

        "### INPUT CONTEXT ###\n"
        f"{text_context}\n"
        f"{image_context if image_context else ''}\n\n"

        "### QUESTION ###\n"
        f"{question}\n\n"

        "### ANSWER ###\n"
        )

        answer_data = await cohere_api_request("generate", {"prompt": final_prompt, "model": "command-r-plus", "max_tokens": 500, "temperature": 0.2})
        answer = answer_data["generations"][0]["text"]
        answers.append(answer)
        qa_results_for_saving.append({"id": i + 1, "question": question, "answer": answer, "text_context": text_context, "image_context": image_context})

    save_session_to_markdown(doc_url, qa_results_for_saving)
    logger.info(f"🎉 All questions processed in {time.time() - start_time:.2f}s.")
    return {"answers": answers}

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Main Web App Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/clear_cache")
def clear_cache():
    global document_cache
    document_cache.clear()
    logger.info("Cache has been cleared.")
    return {"status": "cache cleared"}