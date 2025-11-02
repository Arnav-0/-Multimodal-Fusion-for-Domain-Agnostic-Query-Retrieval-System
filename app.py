
import os
import io
import time
import json
import uuid
import logging
from typing import List, Optional, Dict, Any

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="🧠 Multimodal Q&A Dashboard", page_icon="🧠", layout="wide")

# Initialize session state for API URL
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("UNIFIED_URL", "http://127.0.0.1:8000")

# Logging
logger = logging.getLogger("app")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Styles (small, tasteful)
st.markdown(
    """
    <style>
    .card { background: var(--bg-color); border-radius:10px; padding:12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom:12px }
    .qa-question { font-weight:600; color:#0f172a }
    .qa-answer { margin-top:8px; white-space:pre-wrap }
    .footer { color: #888; font-size:12px; margin-top:24px }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("🧠 Controls")
    
    # Settings Section
    with st.expander("⚙️ Settings", expanded=False):
        st.markdown("### API Configuration")
        new_api_url = st.text_input(
            "API Base URL", 
            value=st.session_state.api_url,
            help="Enter the base URL for your API server (e.g., http://127.0.0.1:8000)",
            key="api_url_input"
        )
        
        if st.button("Update API URL", key="update_api_btn"):
            # Validate URL format
            if new_api_url.strip():
                # Remove trailing slash if present
                cleaned_url = new_api_url.strip().rstrip('/')
                st.session_state.api_url = cleaned_url
                st.success(f"✅ API URL updated to: {cleaned_url}")
                st.info("🔄 Reloading page to apply changes...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please enter a valid URL")
        
        st.caption(f"Current API: `{st.session_state.api_url}`")
        st.markdown("---")
    
    st.caption("Document source and fusion settings")

    src_type = st.radio("Document Source", ("URL", "Upload PDF"))
    doc_url = ""
    upload_path = ""
    if src_type == "URL":
        doc_url = st.text_input("PDF URL", value="")
    else:
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
        if uploaded is not None:
            cache_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "_lf_cache", "ui_uploads")
            os.makedirs(cache_dir, exist_ok=True)
            fname = f"upload_{int(time.time()*1000)}_{uuid.uuid4().hex}.pdf"
            fpath = os.path.join(cache_dir, fname)
            with open(fpath, "wb") as f:
                f.write(uploaded.read())
            upload_path = fpath
            st.success(f"Saved: {os.path.basename(fpath)}")

    fusion = st.selectbox("Fusion Mode", ("late", "early", "hybrid"), index=0)
    include_images = st.checkbox("Include images in answers", value=True)

    st.markdown("---")
    st.markdown("### Sample Questions")
    sample_q = st.selectbox("Quick pick", (
        "Summarize the report",
        "List key findings",
        "Describe the figure on page 3",
        "Show important numbers",
        "Any action items?",
    ), index=0)

    st.markdown("---")
    st.markdown("### Server Health")
    ok_unified = False
    ok_model = False
    device = "?"
    health_error = None
    try:
        # Use the session state API URL for health check
        health_check_url = f"{st.session_state.api_url.rstrip('/')}/hackrx/health"
        h = requests.get(health_check_url, timeout=3)
        if h.ok:
            hj = h.json()
            ok_unified = True
            ok_model = hj.get("model_server", {}).get("ok", False)
            device = hj.get("model_server", {}).get("device", "?")
        else:
            health_error = f"HTTP {h.status_code}"
    except requests.exceptions.RequestException as e:
        health_error = str(e)

    st.markdown(f"Unified API: {'✅' if ok_unified else '❌'}")
    st.markdown(f"Model Server: {'✅' if ok_model else '❌'} ({device})")
    if health_error:
        st.caption(f"{health_error}")

    st.markdown("---")
    if st.button("Run health check now"):
        st.rerun()

# Main layout
st.title("🧠 Multimodal Q&A Dashboard")
cols = st.columns([3, 1])
with cols[1]:
    if st.button("Clear cache & results"):
        if os.path.exists("_lf_cache"):
            try:
                import shutil

                shutil.rmtree("_lf_cache")
                st.success("Cache cleared")
            except Exception as e:
                st.error(f"Could not clear cache: {e}")

# Input area
with st.expander("Ask a question", expanded=True):
    question = st.text_area("Question", value=sample_q, height=120)
    run_btn = st.button("Run Q&A", help="Send request to backend and fetch answers")

# Tabs
tab_answers, tab_debug, tab_preview = st.tabs(["Answers 💡", "Debug Info 📊", "Preview 📄"])

# state
if "results" not in st.session_state:
    st.session_state.results = None
if "last_run" not in st.session_state:
    st.session_state.last_run = {}


def get_active_document() -> str:
    if src_type == "Upload PDF" and upload_path:
        return upload_path
    return doc_url.strip()


@st.cache_data(ttl=3600)
def call_backend(document: str, questions: List[str], fusion_mode: str, include_images_flag: bool, api_base_url: str) -> Dict[str, Any]:
    """Call the backend API with the configured API URL"""
    run_url = f"{api_base_url.rstrip('/')}/hackrx/run"
    payload = {"documents": document, "questions": questions, "fusion": fusion_mode, "include_images": include_images_flag}
    resp = requests.post(run_url, json=payload, timeout=240)
    resp.raise_for_status()
    return resp.json()


# Run flow
if run_btn:
    doc = get_active_document()
    if not doc:
        st.error("Please provide a document URL or upload a PDF before running.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Running Q&A… This may take a while for large documents"):
            try:
                t0 = time.time()
                res = call_backend(doc, [question], fusion, include_images, st.session_state.api_url)
                dt = time.time() - t0
                st.session_state.results = res
                st.session_state.last_run = {"latency": dt, "document": doc, "fusion": fusion}
                st.success(f"Completed in {int(dt)}s")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try again or increase server timeout.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Is the API running?")
            except Exception as e:
                st.error(f"Error: {e}")


# Answers tab
with tab_answers:
    if not st.session_state.results:
        st.info("No results yet. Enter a question and click Run Q&A.")
    else:
        answers = st.session_state.results.get("answers", [])
        debug = st.session_state.results.get("debug", {})
        for i, a in enumerate(answers):
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='qa-question'>Q: {st.session_state.last_run.get('fusion')} - {st.session_state.last_run.get('document')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='qa-answer'>{a}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # download
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download JSON"):
                payload = {"meta": st.session_state.last_run, "results": st.session_state.results}
                st.download_button("Download JSON", json.dumps(payload, indent=2), file_name="qa_results.json", mime="application/json")
        with col2:
            if st.button("Download TXT"):
                text_out = []
                for q, a in zip([question], answers):
                    text_out.append(f"Q: {q}\nA: {a}\n---\n")
                st.download_button("Download TXT", "\n".join(text_out), file_name="qa_results.txt", mime="text/plain")

# Debug tab
with tab_debug:
    if not st.session_state.results:
        st.info("No debug info available until you run a query.")
    else:
        debug = st.session_state.results.get("debug", {})
        latency = st.session_state.last_run.get("latency")
        st.write(f"API Latency: {latency:.2f}s")
        with st.expander("Full debug JSON", expanded=False):
            st.json(debug)
        # show retrieval hits if present
        per_q = debug.get("per_q") or []
        if per_q and isinstance(per_q, list):
            for idx, pq in enumerate(per_q):
                if isinstance(pq, dict):
                    with st.expander(f"Question {idx+1} retrieval details", expanded=False):
                        st.write("Text hits:")
                        text_hits = pq.get("text_hits", [])
                        if isinstance(text_hits, list):
                            for hit in text_hits[:10]:
                                st.write(hit)
                        st.write("Image hits:")
                        image_hits = pq.get("image_hits", [])
                        if isinstance(image_hits, list):
                            for ih in image_hits[:10]:
                                st.write(ih)

# Preview tab
with tab_preview:
    if not st.session_state.results:
        st.info("No preview available until you run a query.")
    else:
        debug = st.session_state.results.get("debug", {})
        per_q = debug.get("per_q") or []
        imgs = []
        if per_q and isinstance(per_q, list) and len(per_q) > 0:
            first_q = per_q[0]
            if isinstance(first_q, dict):
                used_images = first_q.get("used_images", [])
                if isinstance(used_images, list):
                    imgs = used_images
        if not imgs:
            st.info("No images extracted for preview.")
        else:
            cols = st.columns(3)
            for i, p in enumerate(imgs[:9]):
                col = cols[i % 3]
                try:
                    col.image(Image.open(p), caption=os.path.basename(p), use_column_width=True)
                except Exception:
                    col.write(p)
