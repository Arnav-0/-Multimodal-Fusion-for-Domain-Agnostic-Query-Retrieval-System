# How to Restart Servers After Code Changes

## Quick Restart (Recommended)

```powershell
# 1. Stop all servers
.\stop_all.ps1

# 2. Wait 2 seconds for ports to free
Start-Sleep -Seconds 2

# 3. Start everything again
.\start_unified.ps1
```

## Manual Restart

If you have terminals open with running servers:

### Terminal 1 - Model Server (port 8001)
1. Press `Ctrl+C` to stop
2. Run: `python -m uvicorn model_server:app --host 127.0.0.1 --port 8001`

### Terminal 2 - Unified API (port 8000)
1. Press `Ctrl+C` to stop
2. Run: `python -m uvicorn main_api:app --host 127.0.0.1 --port 8000`

### Terminal 3 - Streamlit (port 8501)
1. Press `Ctrl+C` to stop
2. Run: `python -m streamlit run app.py --server.port 8501 --server.headless true`

## Why Restart is Needed

- Python imports modules once at startup
- Code changes in `.py` files don't take effect until the process restarts
- The error you saw (`TypeError: run_latefusion() got an unexpected keyword argument 'include_images'`) means the server is running old code

## Files Changed in Latest Fix

1. ✅ `main_api.py` - Now passes `include_images` to all fusion types
2. ✅ `main_latefusion.py` - Updated signature to accept `include_images`
3. ✅ `main_earlyfusion.py` - Updated signature to accept `include_images`
4. ✅ `main_hybridfusion.py` - Updated signature to accept `include_images`
5. ✅ `app.py` - Fixed endpoint URL and removed footer
6. ✅ All files compiled successfully (no syntax errors)

## After Restart

Your system should work end-to-end with no errors:
- All three fusion modes (late/early/hybrid) accept `include_images` parameter
- Frontend connects to correct `/hackrx/run` endpoint
- Images are used or excluded based on toggle setting
