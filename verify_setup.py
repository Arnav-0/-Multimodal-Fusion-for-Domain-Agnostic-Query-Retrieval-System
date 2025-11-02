"""
System Verification Script
Checks if all dependencies and configurations are correct
"""
import os
import sys

def check_environment():
    print("=" * 80)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 80)
    
    # Check .env file
    print("\n1. Checking .env file...")
    if os.path.exists('.env'):
        print("   ✓ .env file found")
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check critical variables
        gemini_key = os.getenv("GEMINI_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL")
        device = os.getenv("MODEL_DEVICE", "cpu")
        
        if gemini_key:
            print(f"   ✓ GEMINI_API_KEY is set")
        else:
            print("   ✗ GEMINI_API_KEY is missing!")
            
        print(f"   ✓ GEMINI_MODEL: {gemini_model}")
        print(f"   ✓ MODEL_DEVICE: {device}")
    else:
        print("   ✗ .env file not found!")
        return False
    
    # Check Python packages
    print("\n2. Checking Python packages...")
    packages = [
        "fastapi",
        "uvicorn", 
        "torch",
        "sentence_transformers",
        "faiss",
        "fitz",
        "PIL",
        "pytesseract",
        "google.generativeai",
        "numpy",
        "requests"
    ]
    
    missing = []
    for pkg in packages:
        try:
            if pkg == "fitz":
                __import__("fitz")
            elif pkg == "PIL":
                __import__("PIL")
            else:
                __import__(pkg.replace("-", "_"))
            print(f"   ✓ {pkg}")
        except ImportError:
            print(f"   ✗ {pkg} - NOT INSTALLED")
            missing.append(pkg)
    
    if missing:
        print(f"\n   Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
    
    # Check CUDA
    print("\n3. Checking CUDA/GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA is available")
            print(f"   ✓ CUDA Version: {torch.version.cuda}")
            print(f"   ✓ GPU Count: {torch.cuda.device_count()}")
            print(f"   ✓ GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"   ✗ Error checking CUDA: {e}")
    
    # Check FAISS GPU
    print("\n4. Checking FAISS GPU...")
    try:
        import faiss
        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            print(f"   ✓ FAISS GPU available: {num_gpus} GPU(s)")
        else:
            print("   ⚠ FAISS GPU not available (will use CPU)")
    except Exception as e:
        print(f"   ⚠ FAISS GPU check: {e}")
    
    # Check project files
    print("\n5. Checking project files...")
    files = [
        "main_latefusion.py",
        "model_server.py",
        "utils.py",
        "test_api.py"
    ]
    
    for f in files:
        if os.path.exists(f):
            print(f"   ✓ {f}")
        else:
            print(f"   ✗ {f} - MISSING")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    if missing:
        print("\n⚠ Please install missing packages before running the servers.")
        return False
    
    print("\n✓ System is ready!")
    print("\nNext steps:")
    print("1. Start model server: uvicorn model_server:app --port 8001")
    print("2. Start main server: uvicorn main_latefusion:app --port 8000")
    print("3. Or run: .\\start_servers.ps1")
    return True

if __name__ == "__main__":
    try:
        check_environment()
    except Exception as e:
        print(f"\nError during verification: {e}")
        sys.exit(1)
