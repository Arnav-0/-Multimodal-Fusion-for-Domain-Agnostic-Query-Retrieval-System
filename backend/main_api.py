import os
import logging
import time
import requests
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Reuse implementations by importing modules
import main_latefusion as late
import main_earlyfusion as early
import main_hybridfusion as hybrid


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("main_api")

app = FastAPI(title="Unified Fusion QA (Early | Late | Hybrid)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


class RunRequest(BaseModel):
    documents: str
    questions: List[str]
    fusion: str = Field("late", description="one of: late | early | hybrid")
    include_images: Optional[bool] = True


class RunResponse(BaseModel):
    answers: List[str]
    debug: Dict[str, Any]


@app.get("/hackrx/fusions")
def list_fusions():
    return {
        "available": ["late", "early", "hybrid"],
        "unified_endpoint": "/hackrx/run",
        "individual_endpoints": {
            "late": "/hackrx/run_latefusion (only when running main_latefusion:app)",
            "early": "/hackrx/run_earlyfusion (only when running main_earlyfusion:app)",
            "hybrid": "/hackrx/run_hybridfusion (only when running main_hybridfusion:app)",
        },
    }


@app.get("/health")
@app.get("/hackrx/health")
def health():
    """Health check endpoint for load balancers and monitoring"""
    # Model server status
    model_base = os.getenv("MODEL_SERVER_URL", "http://localhost:8001").rstrip("/")
    model_url = f"{model_base}/health"
    logger.info(f"Checking model server health at: {model_url}")
    model = {"ok": False, "device": None, "detail": None}
    try:
        r = requests.get(model_url, timeout=5)
        logger.info(f"Model server response: {r.status_code}")
        if r.ok:
            j = r.json()
            model.update({"ok": True, "device": j.get("device"), "device_id": j.get("device_id")})
        else:
            model["detail"] = f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError as ce:
        logger.warning(f"Model server connection error: {ce}")
        model["detail"] = "Connection refused"
    except requests.exceptions.Timeout:
        logger.warning("Model server timeout")
        model["detail"] = "Timeout"
    except Exception as e:
        logger.warning(f"Model server health check error: {e}")
        model["detail"] = str(e)[:100]

    return {
        "status": "ok" if model.get("ok") else "degraded",
        "service": "unified_fusion_api",
        "time": int(time.time()),
        "unified_endpoint": "/hackrx/run",
        "fusions": ["late", "early", "hybrid"],
        "model_server": model,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "gpu_enabled": os.getenv("MODEL_DEVICE", "cpu") == "cuda"
    }


@app.post("/hackrx/run", response_model=RunResponse)
async def run(body: RunRequest):
    fusion = (body.fusion or "late").lower().strip()
    logger.info(f"Received request for fusion={fusion}, questions={body.questions}")
    try:
        if fusion == "late":
            # Adapt to LFRequest structure
            req = late.LFRequest(documents=body.documents, questions=body.questions, fusion="late")
            # Pass include_images as an extra field
            resp = await late.run_latefusion(req, include_images=body.include_images)
            logger.info(f"Late fusion completed: {len(resp.answers)} answers")
            return RunResponse(answers=resp.answers, debug=resp.debug)
        elif fusion == "early":
            req = early.EFRequest(documents=body.documents, questions=body.questions, fusion="early")
            resp = await early.run_earlyfusion(req, include_images=body.include_images)
            logger.info(f"Early fusion completed: {len(resp.answers)} answers")
            return RunResponse(answers=resp.answers, debug=resp.debug)
        elif fusion == "hybrid":
            req = hybrid.HFRequest(documents=body.documents, questions=body.questions, fusion="hybrid")
            resp = await hybrid.run_hybridfusion(req, include_images=body.include_images)
            logger.info(f"Hybrid fusion completed: {len(resp.answers)} answers")
            return RunResponse(answers=resp.answers, debug=resp.debug)
        else:
            error_msg = f"[Error: unknown fusion '{fusion}' (use late|early|hybrid)]"
            logger.error(error_msg)
            return RunResponse(
                answers=[error_msg],
                debug={"fusion": fusion, "error": "unknown_fusion"}
            )
    except Exception as e:
        logger.exception("Unified run failed for fusion=%s", fusion)
        error_detail = f"{type(e).__name__}: {str(e)}"
        return RunResponse(
            answers=[f"[Error during {fusion} fusion: {error_detail}]"] * len(body.questions),
            debug={"fusion": fusion, "error": error_detail, "error_type": type(e).__name__}
        )
