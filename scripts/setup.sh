#!/bin/bash
set -e

echo "🚀 Setting up LiveKit Meeting Intelligence System..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Backend setup
echo "📦 Setting up Python backend..."
cd backend

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
uv pip install -e .

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit backend/.env with your API keys"
fi

cd ..

# Frontend setup
echo "🎨 Setting up Next.js frontend..."
cd frontend

# Install dependencies
echo "📥 Installing Node dependencies..."
npm install

# Create .env.local if it doesn't exist
if [ ! -f .env.local ]; then
    echo "📝 Creating .env.local file..."
    cat > .env.local << 'EOF'
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/transcripts
NEXT_PUBLIC_API_URL=http://localhost:8000/api
EOF
fi

cd ..

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit backend/.env with your LiveKit and OpenAI API keys"
echo "2. Start LiveKit server (if running locally): livekit-server --dev"
echo "3. Run ./scripts/start-all.sh to start all services"
