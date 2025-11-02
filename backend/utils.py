import os
import re
import time
import json
import hashlib
import logging
import random
import threading
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as gexc

load_dotenv()

# ---------------- Logging ----------------
logger = logging.getLogger("utils")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------- Config ----------------
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001").rstrip("/")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
GEMINI_INCLUDE_IMAGES = os.getenv("GEMINI_INCLUDE_IMAGES", "true").lower() == "true"
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "")
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_BACKOFF_BASE = float(os.getenv("GEMINI_BACKOFF_BASE", "5.0"))  # seconds
GEMINI_BACKOFF_MAX = float(os.getenv("GEMINI_BACKOFF_MAX", "60.0"))
GEMINI_BACKOFF_JITTER = float(os.getenv("GEMINI_BACKOFF_JITTER", "0.5"))
GEMINI_CACHE_PATH = os.getenv("GEMINI_CACHE_PATH", ".gemini_cache")
GEMINI_CACHE_TTL = int(os.getenv("GEMINI_CACHE_TTL", "86400"))  # seconds
GEMINI_CONCURRENCY = int(os.getenv("GEMINI_CONCURRENCY", "2"))
GEMINI_CIRCUIT_FAILS = int(os.getenv("GEMINI_CIRCUIT_FAILS", "6"))
GEMINI_CIRCUIT_RESET = int(os.getenv("GEMINI_CIRCUIT_RESET", "120"))  # seconds

# in-memory circuit breaker state and semaphore
_gemini_semaphore = threading.BoundedSemaphore(GEMINI_CONCURRENCY)
_circuit_lock = threading.Lock()
_circuit_state = {
    "fail_count": 0,
    "opened_until": 0.0,
}

# ensure cache dir exists
try:
    os.makedirs(GEMINI_CACHE_PATH, exist_ok=True)
except Exception:
    pass

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ---------------- Model Server helpers ----------------
def _post_json(path: str, payload: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
    url = f"{MODEL_SERVER_URL}/{path.lstrip('/')}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_text_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return _post_json("/get_text_embeddings", {"texts": texts}).get("embeddings", [])


def get_clip_text_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return _post_json("/get_clip_text_embeddings", {"texts": texts}).get("embeddings", [])


def get_image_embeddings(paths: List[str]) -> List[List[float]]:
    if not paths:
        return []
    return _post_json("/get_image_embeddings", {"paths": paths}).get("embeddings", [])


def rerank_candidates(query: str, candidates: List[str]) -> List[float]:
    if not candidates:
        return []
    return _post_json("/rerank", {"query": query, "candidates": candidates}).get("scores", [])


# ---------------- Gemini helpers (inline images; no uploads/RAG) ----------------
def gemini_generate(prompt: str, image_paths: Optional[List[str]] = None, model: Optional[str] = None) -> str:
    """
    Generate an answer using Gemini. Images are uploaded using the File API.
    """
    if not GEMINI_API_KEY:
        return "[Gemini unavailable: set GEMINI_API_KEY]"

    # Circuit breaker: if open, immediately return an error to avoid hammering the API
    now = time.time()
    with _circuit_lock:
        if _circuit_state.get("opened_until", 0) > now:
            wait = int(_circuit_state["opened_until"] - now)
            logger.warning("Gemini circuit open, refusing call for %ds", wait)
            return f"[Gemini temporarily unavailable (circuit open, retry in {wait}s)]"

    try:
        from PIL import Image as PILImage

        # simple cache lookup
        try:
            key_src = json.dumps({"prompt": prompt, "images": image_paths or [], "model": (model or GEMINI_MODEL)}, sort_keys=True)
            key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
            cache_file = os.path.join(GEMINI_CACHE_PATH, f"{key}.json")
            if os.path.exists(cache_file):
                st = os.path.getmtime(cache_file)
                if time.time() - st < GEMINI_CACHE_TTL:
                    try:
                        with open(cache_file, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        txt = data.get("text")
                        if txt:
                            logger.debug("Gemini cache hit for prompt (model=%s)", model or GEMINI_MODEL)
                            return txt
                    except Exception:
                        # ignore cache errors
                        pass
        except Exception:
            key = None
            cache_file = None

        attempts = 0
        use_model = (model or GEMINI_MODEL)

        # acquire concurrency semaphore
        acquired = _gemini_semaphore.acquire(timeout=30)
        if not acquired:
            logger.warning("Could not acquire Gemini semaphore - too many concurrent requests")
            return "[Gemini busy: too many concurrent requests]"

        try:
            while True:
                try:
                    # instantiate model and build parts
                    mdl = genai.GenerativeModel(use_model)
                    parts: List[Any] = [prompt]
                    if GEMINI_INCLUDE_IMAGES and image_paths:
                        for p in image_paths:
                            if not p:
                                continue
                            p = os.path.abspath(p)
                            if not os.path.exists(p):
                                continue
                            try:
                                img = PILImage.open(p)
                                parts.append(img)
                            except Exception as img_err:
                                logger.warning(f"Could not load image {p}: {img_err}")
                                continue

                    resp = mdl.generate_content(parts)

                    # reset circuit on success
                    with _circuit_lock:
                        _circuit_state["fail_count"] = 0

                    # Handle blocked or empty responses
                    if not resp or not hasattr(resp, 'text'):
                        logger.warning(f"Gemini returned invalid response: {resp}")
                        return "[Gemini returned an invalid response - content may have been blocked]"

                    try:
                        text = resp.text or ""
                    except ValueError as ve:
                        # This happens when content is blocked due to safety filters
                        logger.warning(f"Gemini response blocked or empty: {ve}")
                        if hasattr(resp, 'prompt_feedback'):
                            logger.warning(f"Prompt feedback: {resp.prompt_feedback}")
                        if hasattr(resp, 'candidates') and resp.candidates:
                            logger.warning(f"Candidates: {resp.candidates}")
                        return "[Content blocked by safety filters or no response generated]"

                    if not text.strip():
                        logger.warning("Gemini returned empty text")
                        return "[Gemini returned an empty response]"

                    text = text.strip()

                    # write cache
                    try:
                        if cache_file and key:
                            with open(cache_file, "w", encoding="utf-8") as fh:
                                json.dump({"text": text, "model": use_model, "ts": time.time()}, fh)
                    except Exception:
                        pass

                    return text

                except gexc.ResourceExhausted as ex:
                    # increment attempts and circuit fail counter
                    attempts += 1
                    with _circuit_lock:
                        _circuit_state["fail_count"] = _circuit_state.get("fail_count", 0) + 1
                        if _circuit_state["fail_count"] >= GEMINI_CIRCUIT_FAILS:
                            _circuit_state["opened_until"] = time.time() + GEMINI_CIRCUIT_RESET
                            logger.error("Gemini circuit opened for %ds due to repeated failures", GEMINI_CIRCUIT_RESET)

                    # fallback model switch if configured
                    if attempts > GEMINI_MAX_RETRIES and GEMINI_FALLBACK_MODEL:
                        logger.warning("Gemini quota exceeded; switching to fallback model: %s", GEMINI_FALLBACK_MODEL)
                        use_model = GEMINI_FALLBACK_MODEL
                        attempts = 0  # reset attempts for fallback
                        continue
                    if attempts > GEMINI_MAX_RETRIES and not GEMINI_FALLBACK_MODEL:
                        raise

                    # parse suggested retry delay if present in message
                    msg = str(ex)
                    m = re.search(r"retry in\s+([0-9.]+)s", msg, re.IGNORECASE)
                    # try to parse Retry-After from exception if available
                    retry_after = None
                    try:
                        # genai exceptions sometimes include a _cause with details
                        cause = getattr(ex, '__cause__', None)
                        if cause and hasattr(cause, 'response') and hasattr(cause.response, 'headers'):
                            hdr = cause.response.headers.get('Retry-After')
                            if hdr:
                                retry_after = float(hdr)
                    except Exception:
                        pass

                    delay = GEMINI_BACKOFF_BASE * (2 ** (attempts - 1))
                    if m:
                        try:
                            delay = max(delay, float(m.group(1)))
                        except Exception:
                            pass
                    if retry_after:
                        delay = max(delay, retry_after)

                    # jitter
                    jitter = GEMINI_BACKOFF_JITTER * (random.random() + 0.5)
                    delay = min(delay * (1.0 + jitter), GEMINI_BACKOFF_MAX)

                    logger.warning("Gemini ResourceExhausted: backing off for %.2fs (attempt %d/%d)", delay, attempts, GEMINI_MAX_RETRIES)
                    time.sleep(delay)
                    continue
                except Exception as e:
                    # non-rate-limit error -> increase fail counter and possibly open circuit
                    with _circuit_lock:
                        _gem_fail = _circuit_state.get("fail_count", 0) + 1
                        _circuit_state["fail_count"] = _gem_fail
                        if _gem_fail >= GEMINI_CIRCUIT_FAILS:
                            _circuit_state["opened_until"] = time.time() + GEMINI_CIRCUIT_RESET
                            logger.error("Gemini circuit opened for %ds due to repeated failures", GEMINI_CIRCUIT_RESET)

                    logger.exception("Gemini generation failed")
                    return f"[Gemini error: {e}]"
        finally:
            try:
                _gemini_semaphore.release()
            except Exception:
                pass
    except Exception as e:
        logger.exception("Gemini generation failed")
        return f"[Gemini error: {e}]"
