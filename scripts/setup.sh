#!/bin/bash

# DTS Voice - One-Command Installation Script
# This script installs all dependencies and configures the project for first use

set -e  # Exit on any error

echo "========================================="
echo "🚀 DTS Voice - Installation Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Get project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

print_info "Project directory: $PROJECT_ROOT"
echo ""

# Step 1: Check macOS and Apple Silicon
echo "📋 Step 1: Checking system requirements..."
echo ""

if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This project requires macOS (for MLX framework)"
    exit 1
fi
print_status "Running on macOS"

ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    print_error "This project requires Apple Silicon (M1/M2/M3/M4)"
    print_error "Current architecture: $ARCH"
    exit 1
fi
print_status "Running on Apple Silicon ($ARCH)"
echo ""

# Step 2: Check/Install Homebrew
echo "📋 Step 2: Checking Homebrew..."
echo ""

if ! command -v brew &> /dev/null; then
    print_warning "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add brew to PATH for Apple Silicon
    if [[ -f "/opt/homebrew/bin/brew" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi

    print_status "Homebrew installed successfully"
else
    print_status "Homebrew already installed ($(brew --version | head -n1))"
fi
echo ""

# Step 3: Install Node.js
echo "📋 Step 3: Checking Node.js..."
echo ""

if ! command -v node &> /dev/null; then
    print_warning "Node.js not found. Installing via Homebrew..."
    brew install node
    print_status "Node.js installed successfully"
else
    NODE_VERSION=$(node --version)
    print_status "Node.js already installed ($NODE_VERSION)"
fi

if ! command -v npm &> /dev/null; then
    print_error "npm not found after Node.js installation"
    exit 1
fi
print_status "npm available ($(npm --version))"
echo ""

# Step 4: Install LiveKit
echo "📋 Step 4: Checking LiveKit server..."
echo ""

if ! command -v livekit-server &> /dev/null; then
    print_warning "LiveKit server not found. Installing via Homebrew..."
    brew install livekit
    print_status "LiveKit server installed successfully"
else
    print_status "LiveKit server already installed"
fi
echo ""

# Step 5: Install uv (Python package manager)
echo "📋 Step 5: Checking uv (Python package manager)..."
echo ""

if ! command -v uv &> /dev/null; then
    print_warning "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH (check both possible locations)
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        print_error "uv installation failed. Please restart your shell and run:"
        print_error "source $HOME/.local/bin/env"
        print_error "Then run this setup script again"
        exit 1
    fi

    print_status "uv installed successfully ($(uv --version))"
else
    print_status "uv already installed ($(uv --version))"
fi
echo ""

# Step 6: Install Python 3.13
echo "📋 Step 6: Installing Python 3.13..."
echo ""

# Install Python 3.13 (required for package compatibility)
print_info "Installing Python 3.13 (required for dependencies)..."
uv python install 3.13
print_status "Python 3.13 installed"
echo ""

# Step 7: Backend Setup
echo "📋 Step 7: Setting up Python backend..."
echo ""

cd "$PROJECT_ROOT/backend"

# Check if venv exists and is using correct Python version
if [ -d ".venv" ]; then
    VENV_PYTHON_VERSION=$(.venv/bin/python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    if [[ "$VENV_PYTHON_VERSION" != "3.13" ]]; then
        print_warning "Existing venv uses Python $VENV_PYTHON_VERSION, but we need 3.13"
        print_info "Removing old virtual environment..."
        rm -rf .venv
        print_info "Creating Python 3.13 virtual environment..."
        uv venv --python 3.13
        print_status "Virtual environment recreated with Python 3.13"
    else
        print_status "Virtual environment already exists with Python 3.13"
    fi
else
    print_info "Creating Python 3.13 virtual environment..."
    uv venv --python 3.13
    print_status "Virtual environment created with Python 3.13"
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
print_info "Installing Python dependencies (this may take a few minutes)..."
uv pip install -e .
print_status "Python dependencies installed"

# Create .env file
if [ ! -f .env ]; then
    print_info "Creating .env file with default settings..."
    cat > .env << 'EOF'
# LiveKit Configuration (local dev defaults)
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# OpenAI API Key (REQUIRED - add your key here)
OPENAI_API_KEY=your-openai-api-key-here

# M4 Apple Silicon Performance Optimizations
MLX_METAL_DEVICE_VERBOSE=0
MLX_METAL_CAPTURE_ENABLED=0
MLX_USE_NEURAL_ENGINE=1

# Python multiprocessing optimization for macOS
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Audio Preprocessing (M4 Optimized)
ENABLE_HIGHPASS_FILTER=true
ENABLE_NORMALIZATION=false
HIGHPASS_CUTOFF_HZ=100
NORMALIZATION_TARGET_DB=-3.0

# Audio Quality Validation (M4 Optimized - more permissive)
ENABLE_AUDIO_VALIDATION=true
AUDIO_VALIDATION_RMS_THRESHOLD_DB=-55.0

# Segmentation Settings
SEGMENT_MIN_WORDS=20
SEGMENT_MAX_WORDS=500
SEGMENT_CONTEXT_SIZE=5
EOF
    print_status ".env file created"
    print_warning "IMPORTANT: Edit backend/.env and add your OpenAI API key!"
else
    print_status ".env file already exists"
fi

cd "$PROJECT_ROOT"
echo ""

# Step 8: Frontend Setup
echo "📋 Step 8: Setting up Next.js frontend..."
echo ""

cd "$PROJECT_ROOT/frontend"

# Install dependencies
print_info "Installing Node.js dependencies (this may take a few minutes)..."
npm install
print_status "Node.js dependencies installed"

# Create .env.local
if [ ! -f .env.local ]; then
    print_info "Creating .env.local file..."
    cat > .env.local << 'EOF'
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/transcripts
NEXT_PUBLIC_API_URL=http://localhost:8000/api
EOF
    print_status ".env.local file created"
else
    print_status ".env.local file already exists"
fi

cd "$PROJECT_ROOT"
echo ""

# Step 9: Create data directory
echo "📋 Step 9: Creating data directories..."
echo ""

mkdir -p data/meetings
print_status "Data directories created"
echo ""

# Summary
echo "========================================="
echo "✅ INSTALLATION COMPLETE!"
echo "========================================="
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Configure your OpenAI API key:"
echo "   ${YELLOW}nano backend/.env${NC}"
echo "   (Replace 'your-openai-api-key-here' with your actual key)"
echo ""
echo "2. Start all services:"
echo "   ${GREEN}./start.sh${NC}"
echo ""
echo "3. Open your browser:"
echo "   ${BLUE}http://localhost:3000${NC}"
echo ""
echo "📚 Additional Commands:"
echo "   • Check service status:  ${GREEN}./status.sh${NC}"
echo "   • Stop all services:     ${GREEN}./stop.sh${NC}"
echo "   • View logs:             ${GREEN}tail -f /tmp/agent.log${NC}"
echo ""
echo "========================================="
echo ""
echo "🎤 Ready to start transcribing!"
echo ""
