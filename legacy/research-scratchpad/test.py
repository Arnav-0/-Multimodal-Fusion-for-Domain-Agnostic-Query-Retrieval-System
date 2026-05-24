import requests
import time
import os
import glob

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/hackrx/run"
TEST_DOCUMENT_URL = "https://arxiv.org/pdf/1512.03385.pdf"  # ResNet Paper

TEST_PAYLOAD = {
    "documents": TEST_DOCUMENT_URL,
    "questions": [
        "Based on the diagram in Figure 2, what is the key difference between the Plain Network building block on the left and the Residual building block on the right?",
        "According to Table 1, what is the top-1 error rate (%) of the 34-layer plain model?"
    ]
}

def run_test():
    """Runs a full suite of automated tests on the running API."""
    print("--- Starting Advanced Multimodal Project Test ---")
    
    # === Test 1: First Request (Cache Miss) ===
    print("\n[1] Testing First Request (Cache Miss)...")
    try:
        start_time = time.time()
        response1 = requests.post(API_URL, json=TEST_PAYLOAD, timeout=600)
        response1.raise_for_status()  # Raise an exception for bad status codes
        time1 = time.time() - start_time
        answers1 = response1.json().get("answers", [])
        print(f"✅ First request successful in {time1:.2f} seconds.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Test Failed: Could not connect to the API. Is the server running? Error: {e}")
        return

    # === Test 2: Second Request (Cache Hit & Performance) ===
    print("\n[2] Testing Second Request (Cache Hit)...")
    try:
        start_time = time.time()
        response2 = requests.post(API_URL, json=TEST_PAYLOAD, timeout=60)
        response2.raise_for_status()
        time2 = time.time() - start_time
        print(f"✅ Second request successful in {time2:.2f} seconds.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Test Failed: Could not connect to the API on second attempt. Error: {e}")
        return
        
    # --- Verification ---
    print("\n--- Verifying Results ---")
    
    # 1. Verify Caching Performance
    print("\n[Verification] Checking cache performance...")
    if time2 < (time1 * 0.5): # Check if second request is at least 50% faster
        print(f"✅ PASS: Caching is working. (First request: {time1:.2f}s, Second request: {time2:.2f}s)")
    else:
        print(f"⚠️ FAIL: Caching might not be working. (First request: {time1:.2f}s, Second request: {time2:.2f}s)")

    # 2. Verify Output Saving
    print("\n[Verification] Checking for saved Markdown report...")
    list_of_files = glob.glob("qa_session_*.md")
    if not list_of_files:
        print("❌ FAIL: No Markdown report file found.")
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"✅ PASS: Found saved report: {latest_file}")

    # 3. Verify Answer Content (Manual Check)
    print("\n[Verification] Please manually verify the answers:")
    print("-" * 50)
    print(f"Question 1 (Visual): {TEST_PAYLOAD['questions'][0]}")
    print(f"--> Answer 1: {answers1[0] if len(answers1) > 0 else 'N/A'}")
    print("\n* Is the answer about the 'shortcut' or 'skip connection' in the diagram?")
    print("-" * 20)
    print(f"Question 2 (Specific Data): {TEST_PAYLOAD['questions'][1]}")
    print(f"--> Answer 2: {answers1[1] if len(answers1) > 1 else 'N/A'}")
    print("\n* Does the answer contain the number '75.34'?")
    print("-" * 50)


if __name__ == "__main__":
    run_test()