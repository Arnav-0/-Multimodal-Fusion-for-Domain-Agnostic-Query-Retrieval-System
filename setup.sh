#!/bin/bash

# Multimodal Document Q&A System - Quick Start Script for Linux/Mac

echo "================================================================================"
echo "  Multimodal Document Q&A System - Quick Start"
echo "================================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${YELLOW}[1/6] Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}  ✓ Python $PYTHON_VERSION found${NC}"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}  ✓ Python $PYTHON_VERSION found${NC}"
    PYTHON_CMD="python"
else
    echo -e "${RED}  ✗ Python not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}[2/6] Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}  ⚠ Virtual environment already exists${NC}"
    read -p "  Recreate? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf .venv
        $PYTHON_CMD -m venv .venv
        echo -e "${GREEN}  ✓ Virtual environment recreated${NC}"
    else
        echo -e "${GREEN}  ✓ Using existing virtual environment${NC}"
    fi
else
    $PYTHON_CMD -m venv .venv
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}[3/6] Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}  ✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}[4/6] Upgrading pip...${NC}"
pip install --upgrade pip -q
echo -e "${GREEN}  ✓ Pip upgraded${NC}"

# Install dependencies
echo -e "\n${YELLOW}[5/6] Installing dependencies...${NC}"
echo -e "${NC}  This may take several minutes...${NC}"

# Check for CUDA
if $PYTHON_CMD -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${YELLOW}  ⚡ CUDA detected - Installing GPU-enabled packages${NC}"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
fi

pip install -r requirements.txt -q
echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# Setup environment file
echo -e "\n${YELLOW}[6/6] Setting up environment file...${NC}"
if [ -f ".env" ]; then
    echo -e "${YELLOW}  ⚠ .env file already exists${NC}"
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}  ✓ Created .env from template${NC}"
        echo -e "${YELLOW}  ⚠ IMPORTANT: Edit .env and add your GEMINI_API_KEY${NC}"
    else
        echo -e "${YELLOW}  ⚠ .env.example not found${NC}"
    fi
fi

# Summary
echo ""
echo "================================================================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "================================================================================"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Edit .env and add your GEMINI_API_KEY"
echo "  2. Start servers:"
echo "     Terminal 1: python -m uvicorn model_server:app --host 127.0.0.1 --port 8001"
echo "     Terminal 2: python -m uvicorn main_api:app --host 127.0.0.1 --port 8000"
echo "     Terminal 3: python -m streamlit run app.py --server.port 8501"
echo "  3. Open: http://localhost:8501"
echo ""
echo -e "${YELLOW}For detailed instructions, see:${NC}"
echo "  - README.md"
echo "  - STARTUP_GUIDE.md"
echo ""
echo -e "${YELLOW}To verify setup, run:${NC}"
echo "  python verify_setup.py"
echo ""
