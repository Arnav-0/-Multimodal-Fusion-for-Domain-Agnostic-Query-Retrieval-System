# Multimodal Document Q&A System - Setup Script
# This script automates the initial setup process

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  Multimodal Document Q&A System - Setup" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -eq 3 -and $minor -ge 9) {
            Write-Host "  ✓ Python $major.$minor found" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Python 3.9+ required. Found: $pythonVersion" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "  ✗ Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host "`n[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "  ⚠ Virtual environment already exists" -ForegroundColor Yellow
    $response = Read-Host "  Recreate? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Remove-Item -Path ".venv" -Recurse -Force
        python -m venv .venv
        Write-Host "  ✓ Virtual environment recreated" -ForegroundColor Green
    } else {
        Write-Host "  ✓ Using existing virtual environment" -ForegroundColor Green
    }
} else {
    python -m venv .venv
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

# Step 3: Activate virtual environment
Write-Host "`n[3/6] Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"
Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green

# Step 4: Upgrade pip
Write-Host "`n[4/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip -q
Write-Host "  ✓ Pip upgraded" -ForegroundColor Green

# Step 5: Install dependencies
Write-Host "`n[5/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Gray

# Check for GPU
try {
    $cudaCheck = python -c "import torch; print(torch.cuda.is_available())" 2>&1
    if ($cudaCheck -match "True") {
        Write-Host "  ⚡ CUDA detected - Installing GPU-enabled packages" -ForegroundColor Cyan
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
    }
} catch {
    Write-Host "  ℹ Installing CPU-only packages" -ForegroundColor Gray
}

pip install -r requirements.txt -q
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# Step 6: Setup environment file
Write-Host "`n[6/6] Setting up environment file..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  ⚠ .env file already exists" -ForegroundColor Yellow
} else {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "  ✓ Created .env from template" -ForegroundColor Green
        Write-Host "  ⚠ IMPORTANT: Edit .env and add your GEMINI_API_KEY" -ForegroundColor Yellow
    } else {
        Write-Host "  ⚠ .env.example not found" -ForegroundColor Yellow
    }
}

# Summary
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Edit .env and add your GEMINI_API_KEY" -ForegroundColor White
Write-Host "  2. Run: .\start_unified.ps1" -ForegroundColor White
Write-Host "  3. Open: http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions, see:" -ForegroundColor Yellow
Write-Host "  - README.md" -ForegroundColor White
Write-Host "  - STARTUP_GUIDE.md" -ForegroundColor White
Write-Host ""
Write-Host "To verify setup, run:" -ForegroundColor Yellow
Write-Host "  python verify_setup.py" -ForegroundColor White
Write-Host ""
