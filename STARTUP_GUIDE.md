# STARTUP & TROUBLESHOOTING GUIDE

## Quick Start (Recommended)

1. **Stop all existing servers:**
   ```powershell
   .\stop_all.ps1
   ```

2. **Start all servers:**
   ```powershell
   .\start_unified.ps1
   ```

3. **Test connections:**
   ```powershell
   python test_connection.py
   ```

4. **Open Streamlit UI:**
   - Navigate to: http://localhost:8501

---

## What Each Server Does

| Server | Port | Purpose | Endpoint |
|--------|------|---------|----------|
| **Model Server** | 8001 | Embeddings (E5, CLIP) + CrossEncoder reranking | http://localhost:8001/health |
| **Unified API** | 8000 | Main API - routes to Late/Early/Hybrid fusion | http://localhost:8000/hackrx/health |
| **Streamlit UI** | 8501 | Chat interface for document Q&A | http://localhost:8501 |

---

## Common Issues & Fixes

### Issue: "Port already in use" or "Errno 10048"

**Problem:** Another instance is running on the same port

**Fix:**
```powershell
.\stop_all.ps1
# Wait 3 seconds
.\start_unified.ps1
```

### Issue: Model Server shows "device: cpu" instead of "cuda"

**Problem:** CUDA environment not set or PyTorch not seeing GPU

**Fix:**
1. Check CUDA availability:
   ```powershell
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

2. If False, reinstall PyTorch with CUDA:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Restart servers:
   ```powershell
   .\stop_all.ps1
   .\start_unified.ps1
   ```

### Issue: Streamlit shows "Model Server: DOWN"

**Problem:** Model server not starting or connection issue

**Fix:**
1. Check if model server is running:
   ```powershell
   netstat -ano | findstr ":8001"
   ```

2. Test directly:
   ```powershell
   curl http://localhost:8001/health
   ```

3. If not responding, check the Model Server window for errors
   - Look for missing dependencies
   - Check memory issues
   - Verify models can download

### Issue: Streamlit shows "Unified API: DOWN"

**Problem:** Main API server not starting

**Fix:**
1. Check if unified API is running:
   ```powershell
   netstat -ano | findstr ":8000"
   ```

2. Test directly:
   ```powershell
   curl http://localhost:8000/hackrx/health
   ```

3. Check the Unified API window for import errors or missing modules

### Issue: Frontend can't process documents

**Problem:** Connection between services broken

**Fix:**
1. Run connection test:
   ```powershell
   python test_connection.py
   ```

2. Check all health badges in Streamlit sidebar:
   - Both should show **OK** in green
   - Model Server should show **cuda** device

3. If any show DOWN:
   ```powershell
   .\stop_all.ps1
   # Wait for all processes to stop
   .\start_unified.ps1
   # Wait 15 seconds for all to start
   python test_connection.py
   ```

### Issue: Gemini API errors (429, quota exceeded)

**Problem:** Free tier limit reached (50 requests/day)

**Fix:**
1. Configure fallback model in `.env`:
   ```
   GEMINI_FALLBACK_MODEL=gemini-1.5-flash-8b
   ```

2. Or wait 24 hours for quota reset

3. Or upgrade to paid Gemini API tier

---

## Verifying Everything Works

### 1. Check Health Endpoints

```powershell
# Model Server
curl http://localhost:8001/health
# Should return: {"status":"ok","device":"cuda","device_id":0}

# Unified API
curl http://localhost:8000/hackrx/health
# Should show model_server.ok = true, device = "cuda"

# Available fusions
curl http://localhost:8000/hackrx/fusions
# Should list: late, early, hybrid
```

### 2. Test API Call

```powershell
python test_connection.py
```

Should pass all 4 tests:
- ✓ Model Server health
- ✓ Unified API health  
- ✓ Fusion modes list
- ✓ Sample API call

### 3. Test Streamlit UI

1. Open http://localhost:8501
2. Check sidebar health badges:
   - **Unified API: OK** (green)
   - **Model Server: OK (cuda)** (green)
3. Try a preset question
4. Upload a PDF or use default URL
5. Should get answer with images

---

## Manual Startup (If Script Fails)

### Terminal 1 - Model Server
```powershell
cd "d:\Final project"
.\.venv\Scripts\Activate.ps1
$env:MODEL_DEVICE="cuda"
$env:MODEL_DEVICE_ID="0"
python -m uvicorn model_server:app --host 127.0.0.1 --port 8001
```

### Terminal 2 - Unified API
```powershell
cd "d:\Final project"
.\.venv\Scripts\Activate.ps1
python -m uvicorn main_api:app --host 127.0.0.1 --port 8000
```

### Terminal 3 - Streamlit
```powershell
cd "d:\Final project"
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py --server.port 8501
```

---

## Environment Variables (.env)

Key settings you can modify:

```properties
# GPU Settings
MODEL_DEVICE=cuda          # or "cpu" if no GPU
MODEL_DEVICE_ID=0          # GPU index
FAISS_USE_GPU=true         # Use GPU for vector search

# Gemini API
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_FALLBACK_MODEL=gemini-1.5-flash-8b  # Optional fallback

# Retrieval Settings
K_TEXT=8                   # Top-k text chunks
K_IMAGE=8                  # Top-k images
ENABLE_OCR=true           # OCR on images

# Fusion Weights (Hybrid mode)
HYBRID_W_FUSED=1.0
HYBRID_W_TEXT=0.7
HYBRID_W_IMAGE=0.7
```

---

## Getting Help

1. **Check server windows** for error messages
2. **Run test script:** `python test_connection.py`
3. **Check health endpoints** in browser:
   - http://localhost:8001/health
   - http://localhost:8000/hackrx/health
4. **View Streamlit sidebar** for real-time status badges

---

## Success Checklist

- [ ] All servers start without errors
- [ ] Model Server shows device: "cuda"
- [ ] Health endpoints return OK status
- [ ] Test script passes all 4 tests
- [ ] Streamlit UI shows both badges as OK (green)
- [ ] Can ask questions and get answers
- [ ] Images display in chat when relevant
