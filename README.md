# LiveKit Meeting Intelligence System

Real-time meeting transcription and intelligence system powered by LiveKit, MLX Whisper, and OpenAI.

## Overview

This project provides:
- **Real-time transcription** using MLX Whisper (optimized for Apple Silicon)
- **Production-grade audio processing** with high-pass filtering, normalization, and quality validation
- **Voice Activity Detection** using Silero VAD with balanced production settings
- **LLM-powered insights** via OpenAI GPT-4 for extracting key proposals, risks, and action items
- **Modern web interface** built with Next.js showing live transcripts and AI-generated nudges

## Architecture

### Audio Processing Pipeline

```
LiveKit Room Audio (48kHz)
  ↓
Audio Preprocessing
  ├─ High-pass filter (80Hz) - removes rumble/AC hum
  ├─ Peak normalization (-3dB) - consistent levels
  └─ Quality validation - prevents hallucinations
  ↓
Silero VAD (Production Settings)
  ├─ activation_threshold: 0.4
  ├─ min_speech_duration: 0.3s
  ├─ min_silence_duration: 0.6s
  └─ Detects speech boundaries
  ↓
MLX Whisper (whisper-large-v3-turbo)
  ├─ Resample to 16kHz
  ├─ Transcribe speech segment
  └─ Return text + confidence
  ↓
LiveKit Data Channel → Frontend Display
```

### Three-Service Architecture

1. **Production Agent** (`backend/agent/production_agent.py`)
   - Real-time audio preprocessing
   - VAD-based speech detection
   - MLX Whisper transcription
   - Quality validation (prevents hallucinations)

2. **FastAPI Server** (`backend/api/server.py`)
   - Token generation for participants
   - Automatic agent dispatch
   - REST API for LLM insights
   - WebSocket for real-time updates

3. **Next.js Frontend** (`frontend/`)
   - LiveKit room connection
   - Real-time transcript display
   - AI-powered nudges panel
   - Export functionality

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+**
- **Node.js 18+** and npm
- **uv** (Python package manager)
- **LiveKit server** (local or cloud)

## Setup

### 1. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables in `.env`:
```env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
OPENAI_API_KEY=your-openai-api-key

# Optional: Audio preprocessing (defaults shown)
ENABLE_HIGHPASS_FILTER=true
ENABLE_NORMALIZATION=true
HIGHPASS_CUTOFF_HZ=80
NORMALIZATION_TARGET_DB=-3.0

# Optional: Audio validation (prevents hallucinations)
ENABLE_AUDIO_VALIDATION=true
AUDIO_VALIDATION_RMS_THRESHOLD_DB=-55.0
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

Create `.env.local`:
```env
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/transcripts
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### 4. LiveKit Server

**Local Development (Recommended)**
```bash
# Install LiveKit server
brew install livekit

# Start in dev mode
livekit-server --dev
```

**LiveKit Cloud**
Sign up at [livekit.io](https://livekit.io) and use your cloud credentials.

## Running the Application

Start all three services in separate terminals:

**Terminal 1: LiveKit Server**
```bash
livekit-server --dev
```

**Terminal 2: FastAPI Server**
```bash
cd backend
source .venv/bin/activate
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 3: Production Agent**
```bash
cd backend
source .venv/bin/activate
python -m agent.production_agent dev
```

**Terminal 4: Next.js Frontend**
```bash
cd frontend
npm run dev
```

Access the application at: **http://localhost:3000**

## Usage

1. **Open the web interface** at http://localhost:3000
2. **Click "Mic On"** to enable your microphone
3. **Start speaking** - The agent detects speech using VAD
4. **View live transcripts** - Transcriptions appear in real-time
5. **See AI insights** - Nudges categorize content into:
   - Key Proposals
   - Delivery Risks
   - Action Items
6. **Export data** - Click "Export" to download meeting JSON

## Audio Processing Features

### Preprocessing
- **High-pass filter (80Hz)**: Removes low-frequency rumble and AC hum
- **Peak normalization (-3dB)**: Ensures consistent audio levels
- **Real-time processing**: <5ms latency per frame

### Quality Validation
Prevents Whisper hallucinations by rejecting:
- Empty or corrupted audio
- Near-silent audio (below -55dB RMS)
- Stuck buffers (constant signal)
- Excessive clipping (>3%)
- Too-short segments (<0.3s)

### VAD Settings (Production)
- **activation_threshold: 0.4** - Balanced sensitivity
- **min_speech_duration: 0.3s** - Reduces false positives
- **min_silence_duration: 0.6s** - Reliable speech end detection

## Whisper Model Configuration

Default model: `whisper-large-v3-turbo` (recommended for real-time)

To change model, edit `backend/agent/config.py`:
```python
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"  # Fast, good accuracy
# Alternative: "mlx-community/whisper-large-v3"  # Slower, maximum accuracy
```

## API Endpoints

### WebSocket
- `ws://localhost:8000/ws/transcripts` - Real-time transcript stream

### REST API
- `POST /api/livekit/token` - Generate access token and dispatch agent
- `POST /api/nudge` - Generate AI insights from transcripts
- `GET /api/meetings` - List all meeting sessions
- `GET /api/meetings/{id}` - Get meeting details

## Troubleshooting

### No Transcripts Appearing

**Check microphone permissions**:
1. Click lock icon in browser address bar
2. Ensure microphone is allowed

**Verify microphone selection**:
1. Open System Settings → Sound → Input
2. Confirm correct microphone is selected (not BlackHole or virtual device)
3. Verify input level meter moves when speaking

**Check agent logs**:
- Look for "Audio is near-silent" warnings
- Verify "Speech started" / "Speech ended" messages appear
- Check for "Audio validation passed" messages

### MLX Whisper Not Loading

**Solution**: Ensure you're on Apple Silicon:
```bash
uname -m  # Should show "arm64"
uv pip install mlx-whisper
```

### LiveKit Connection Failed

**Solution**: Verify LiveKit server is running:
```bash
ps aux | grep livekit-server
# Should show: livekit-server --dev
```

Check credentials in `.env` match (devkey/secret for local dev).

### WebSocket Disconnected

**Solution**: Ensure FastAPI server is running:
```bash
ps aux | grep uvicorn
# Should show: python -m uvicorn api.server:app
```

### Agent Not Joining Room

**Solution**: Verify all services running:
1. LiveKit server (port 7880)
2. API server (port 8000)
3. Production agent (check logs for "registered worker")

## Technologies Used

### Backend
- **LiveKit Agents SDK** - Real-time audio processing
- **Silero VAD** - Voice activity detection
- **MLX Whisper** - Apple Silicon optimized transcription
- **FastAPI** - High-performance async API
- **OpenAI GPT-4** - AI-powered insights
- **SciPy** - Audio signal processing

### Frontend
- **Next.js 14+** - React framework
- **TypeScript** - Type-safe development
- **Tailwind CSS v4** - Styling
- **LiveKit Client SDK** - Room connection
- **Recharts** - Data visualization
- **Lucide React** - Icons

## Performance

- **Transcription latency**: ~1-2s (MLX Whisper Turbo on M4)
- **Audio preprocessing**: <5ms per frame
- **VAD detection**: Real-time
- **Model loading**: ~5s (first room join, cached thereafter)

## Project Structure

```
livekit-whisper/
├── backend/
│   ├── agent/
│   │   ├── production_agent.py    # Production agent with preprocessing
│   │   ├── audio_preprocessing.py # Audio filter/normalize/validate
│   │   ├── transcription.py       # MLX Whisper integration
│   │   └── config.py             # Configuration settings
│   └── api/
│       ├── server.py             # FastAPI server
│       ├── llm_service.py        # OpenAI integration
│       └── storage.py            # JSON storage
├── frontend/
│   ├── app/                      # Next.js app directory
│   ├── components/               # React components
│   ├── hooks/                    # React hooks
│   └── lib/                      # TypeScript types
├── data/                         # Meeting data storage
├── CLAUDE.md                     # Claude Code instructions
└── README.md                     # This file
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- LiveKit for the real-time communication platform
- OpenAI for GPT-4 language model
- Apple for MLX framework
- Anthropic for Claude AI development assistance
