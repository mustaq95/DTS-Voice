# Setup Guide

## Quick Start

### 1. Run Automated Setup

```bash
./scripts/setup.sh
```

This script will:
- Install `uv` if not present
- Create Python virtual environment
- Install all Python dependencies
- Install Node.js dependencies
- Create environment variable templates

### 2. Configure API Keys

Edit `backend/.env`:
```env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret
OPENAI_API_KEY=your-openai-api-key
```

### 3. Start LiveKit Server

**Local Development:**
```bash
brew install livekit
livekit-server --dev
```

**Or use LiveKit Cloud** - Get credentials from https://livekit.io

### 4. Start Application

**Option A: Separate Terminals (Recommended for Development)**

Terminal 1 - Backend:
```bash
./scripts/start-backend.sh
```

Terminal 2 - Frontend:
```bash
./scripts/start-frontend.sh
```

**Option B: All Services**
```bash
# Start backend in background
./scripts/start-backend.sh &

# Start frontend
./scripts/start-frontend.sh
```

### 5. Open Application

Navigate to: **http://localhost:3000**

## Manual Setup

If you prefer manual setup:

### Backend

```bash
cd backend

# Create venv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Configure
cp .env.example .env
# Edit .env with your credentials

# Start FastAPI server
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start agent
python -m agent.main
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure
cp .env.local.example .env.local
# Edit if needed

# Start dev server
npm run dev
```

## Verification

1. **Backend Health**: Visit http://localhost:8000
   - Should see: `{"status":"ok",...}`

2. **Frontend**: Visit http://localhost:3000
   - Should see the meeting UI

3. **WebSocket**: Check browser console
   - Should see "WebSocket connected" (when backend is running)

## Troubleshooting

### Python Dependencies

If `mlx-whisper` fails to install:
```bash
# Ensure you're on Apple Silicon
uname -m  # Should show: arm64

# Update pip
uv pip install --upgrade pip setuptools wheel

# Try again
uv pip install mlx-whisper
```

### Node Dependencies

If npm install fails:
```bash
# Clear cache
rm -rf node_modules package-lock.json
npm cache clean --force

# Reinstall
npm install
```

### LiveKit Connection

If agent can't connect:
1. Verify LiveKit server is running: `livekit-server --dev`
2. Check credentials in `backend/.env`
3. Ensure port 7880 is not in use

### Port Conflicts

If ports are in use:
- **Frontend (3000)**: Edit `package.json` → change port in dev script
- **Backend (8000)**: Edit start scripts → change `--port 8000`
- **LiveKit (7880)**: Edit `.env` → change `LIVEKIT_URL`

## Next Steps

After setup:
1. Test with sample audio
2. Join a LiveKit room
3. Speak and verify transcription
4. Check nudges generation

See [ARCHITECTURE.md](./ARCHITECTURE.md) for system details.
