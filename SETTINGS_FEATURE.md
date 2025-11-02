# ⚙️ Settings Page Feature - API URL Configuration

## Overview
Added a **Settings** section in the Streamlit frontend sidebar where users can configure their API base URL. This allows flexible deployment and easy switching between different API servers.

---

## 🎯 What Was Added

### 1. Settings Section in Sidebar
- New expandable **"⚙️ Settings"** section at the top of the sidebar
- Clean UI with text input for API URL
- "Update API URL" button to apply changes
- Current API URL displayed for reference

### 2. Dynamic API URL Management
- API URL stored in `st.session_state.api_url`
- Initialized from environment variable `UNIFIED_URL` or defaults to `http://127.0.0.1:8000`
- Updates applied immediately with automatic page reload

### 3. All API Calls Updated
- Health check endpoint uses dynamic URL
- Backend Q&A calls use dynamic URL
- URL validation and formatting (removes trailing slashes)

---

## 📍 How to Use

### Step 1: Open Settings
1. Navigate to http://localhost:8501 in your browser
2. Look for **"⚙️ Settings"** at the top of the left sidebar
3. Click to expand the settings section

### Step 2: Update API URL
1. Enter your API base URL in the text input field
   - Examples:
     - Local: `http://127.0.0.1:8000`
     - Remote: `http://192.168.1.100:8000`
     - Cloud: `https://api.yourserver.com`
2. Click **"Update API URL"** button
3. Page will automatically reload with the new settings

### Step 3: Verify Connection
- Check the **"Server Health"** section below
- You should see:
  - ✅ Unified API
  - ✅ Model Server (with device info)
- If you see ❌, verify your API URL is correct and the server is running

---

## 🔧 Technical Details

### Code Changes in `app.py`

#### 1. Session State Initialization
```python
# Initialize session state for API URL
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("UNIFIED_URL", "http://127.0.0.1:8000")
```

#### 2. Settings UI
```python
with st.expander("⚙️ Settings", expanded=False):
    st.markdown("### API Configuration")
    new_api_url = st.text_input(
        "API Base URL", 
        value=st.session_state.api_url,
        help="Enter the base URL for your API server",
        key="api_url_input"
    )
    
    if st.button("Update API URL", key="update_api_btn"):
        if new_api_url.strip():
            cleaned_url = new_api_url.strip().rstrip('/')
            st.session_state.api_url = cleaned_url
            st.success(f"✅ API URL updated to: {cleaned_url}")
            st.rerun()
```

#### 3. Dynamic Health Check
```python
health_check_url = f"{st.session_state.api_url.rstrip('/')}/hackrx/health"
h = requests.get(health_check_url, timeout=3)
```

#### 4. Dynamic Backend Calls
```python
def call_backend(..., api_base_url: str) -> Dict[str, Any]:
    run_url = f"{api_base_url.rstrip('/')}/hackrx/run"
    resp = requests.post(run_url, json=payload, timeout=240)
```

---

## 🌐 Use Cases

### 1. Local Development
```
Default: http://127.0.0.1:8000
```
- Backend running locally on default port

### 2. Network Deployment
```
Example: http://192.168.1.50:8000
```
- Backend running on another machine in local network
- Update API URL to point to that machine's IP

### 3. Cloud Deployment
```
Example: https://api.mycompany.com
```
- Backend deployed to cloud server
- Use HTTPS URL with proper domain

### 4. Different Ports
```
Example: http://127.0.0.1:9000
```
- Backend running on non-default port
- Simply change port number in settings

---

## ✅ Benefits

1. **Flexibility**: No need to restart Streamlit to change API endpoint
2. **Easy Testing**: Quick switching between dev, staging, and production APIs
3. **Multi-Environment**: Same frontend can work with different backends
4. **User-Friendly**: Visual interface for configuration (no code/config file editing)
5. **Persistent**: Settings retained during the session
6. **Validation**: Automatic URL formatting and error handling

---

## 🔍 Features

### URL Validation
- Automatically removes trailing slashes
- Checks for empty input
- Displays current active URL

### Visual Feedback
- ✅ Success message when URL is updated
- 🔄 Loading indicator during page reload
- ❌ Error message for invalid input
- Current URL always visible

### Health Monitoring
- Real-time server status check
- Shows both Unified API and Model Server status
- Device information (CPU/GPU) displayed
- Manual health check button available

---

## 📝 Example Workflow

### Scenario: Switching from Local to Remote Server

1. **Current State**: Using local API at `http://127.0.0.1:8000`
2. **Action**: Open Settings, enter `http://192.168.1.100:8000`
3. **Click**: "Update API URL"
4. **Result**: 
   - Settings saved
   - Page reloads
   - Health check runs against new URL
   - All Q&A requests now go to new server

### Scenario: Testing Different Deployments

```
Morning:   http://127.0.0.1:8000        (Local dev)
Afternoon: http://staging.server:8000   (Staging tests)
Evening:   https://prod.api.com         (Production demo)
```

Simply update the URL in settings - no need to modify code or restart!

---

## 🛡️ Error Handling

### Connection Errors
- If health check fails, shows ❌ with error message
- Backend calls show user-friendly error: "Could not connect to backend. Is the API running?"

### Timeout Errors
- Health check has 3-second timeout
- Backend calls have 240-second timeout
- Clear error messages for users

### Invalid URLs
- Empty URL validation before updating
- Error message: "❌ Please enter a valid URL"

---

## 🔄 Session Persistence

- Settings stored in `st.session_state`
- Persists throughout the browser session
- Resets when browser tab is closed
- Can be overridden via environment variable `UNIFIED_URL`

---

## 🎨 UI Design

### Location
- **Sidebar** → Top section → **"⚙️ Settings"** expander
- Collapsed by default to keep UI clean
- Easy to find with gear icon emoji

### Components
1. Text input with current URL as default value
2. Update button with clear label
3. Success/error message feedback
4. Current URL caption for reference
5. Separator line to distinguish from other controls

---

## 📦 No Additional Dependencies

This feature uses only existing Streamlit components:
- `st.session_state` for state management
- `st.text_input` for URL input
- `st.button` for update action
- `st.rerun()` for page refresh
- No new packages required!

---

## 🚀 Ready to Use

The feature is **immediately available** after restarting the Streamlit server:

```powershell
# If Streamlit is running, restart to load changes
# The server should auto-reload if file watching is enabled
```

Open http://localhost:8501 and look for **⚙️ Settings** in the sidebar!

---

**Date**: 2025-11-02  
**Status**: ✅ Implemented & Ready  
**Impact**: HIGH - Enables flexible API configuration without code changes
