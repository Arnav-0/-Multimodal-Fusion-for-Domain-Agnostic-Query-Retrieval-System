# File: utils.py
import fitz
from PIL import Image
import io
import requests
import base64
from typing import List, Dict, Any
import numpy as np
from docx import Document
import logging

logger = logging.getLogger(__name__)
MODEL_SERVER_URL = "http://127.0.0.1:8001"

# --- Document Processing ---
def load_pdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return ""

def load_docx(path: str) -> str:
    try:
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error reading DOCX file: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    if not text: return []
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    if not pdf_path: return []
    doc = fitz.open(pdf_path)
    images_data = []
    for page_num in range(len(doc)):
        for img in doc.get_page_images(page_num):
            xref = img[0]
            base_image = doc.extract_image(xref)
            try:
                image = Image.open(io.BytesIO(base_image["image"]))
                if image.mode != 'RGB': image = image.convert('RGB')
                images_data.append({"image": image, "page_number": page_num + 1})
            except Exception as e:
                logger.warning(f"Could not open image on page {page_num + 1}. Error: {e}")
    return images_data

# --- Model Server Clients ---
def get_image_embeddings_from_server(images_data: List[Dict[str, Any]]) -> np.ndarray | None:
    if not images_data: return None
    embeddings = []
    for item in images_data:
        pil_image = item["image"]
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        try:
            response = requests.post(f"{MODEL_SERVER_URL}/get_image_embedding", json={"image_b64": img_str})
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        except requests.RequestException as e:
            logger.error(f"Could not connect to model server for image embeddings: {e}")
            return None
    return np.array(embeddings).astype('float32')

def get_text_embedding_from_server(texts: List[str]) -> np.ndarray | None:
    try:
        response = requests.post(f"{MODEL_SERVER_URL}/get_text_embedding", json={"texts": texts})
        response.raise_for_status()
        return np.array(response.json()["embeddings"]).astype('float32')
    except requests.RequestException as e:
        logger.error(f"Could not connect to model server for text embeddings: {e}")
        return None

def get_visual_explanation_from_server(image: Image.Image, question: str) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    try:
        response = requests.post(f"{MODEL_SERVER_URL}/get_visual_explanation", json={"image_b64": img_str, "question": question})
        response.raise_for_status()
        return response.json()["explanation"]
    except requests.RequestException as e:
        logger.error(f"Could not connect to model server for explanation: {e}")
        return "Error: Could not get visual explanation from model server."