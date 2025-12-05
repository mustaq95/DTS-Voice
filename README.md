# DTS Voice - Real-time Meeting Transcription & Intelligence

AI-powered meeting transcription system with real-time topic segmentation and intelligent nudges. Optimized for Apple Silicon (M1/M2/M3/M4).

## Overview

**DTS Voice** provides:
- **Real-time transcription** using MLX Whisper (Apple Silicon optimized)
- **Topic segmentation** with MLX-LM classifier (Qwen2.5-1.5B)
- **AI-powered nudges** via GPT-4.1 (key proposals, delivery risks, action items)
- **Modern Next.js interface** with live transcripts and segment-based UI
- **Production-grade audio processing** with VAD, preprocessing, and quality validation

## Quick Start

### One-Command Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/mustaq95/DTS-Voice.git
cd DTS-Voice

# 2. Run the setup script (installs all dependencies)
./scripts/setup.sh

# 3. Configure your API keys
# Edit backend/.env and add your OpenAI API key

# 4. Start all services
./start.sh
```

That's it! Open http://localhost:3000 in your browser and start transcribing.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4) - Required for MLX
- **Homebrew** - [Install from brew.sh](https://brew.sh)
- **Git** - Included with macOS
- Internet connection for initial setup

## Architecture

### Three-Service Model

1. **LiveKit Production Agent** (`backend/agent/production_agent.py`)
   - Audio resampling (48kHz → 16kHz) for 3x faster VAD
   - Real-time audio preprocessing (high-pass filter, normalization)
   - Silero VAD with M4-optimized settings (continuous speech capture)
   - MLX Whisper transcription (whisper-large-v3-turbo)
   - Non-blocking transcription - VAD continues during processing
   - Topic segmentation via MLX-LM classifier (separate process)

2. **FastAPI Server** (`backend/api/server.py`)
   - LiveKit access token generation
   - Automatic agent dispatch to rooms
   - REST API for nudge generation
   - System redeploy endpoint

3. **Next.js Frontend** (`frontend/app/page.tsx`)
   - LiveKit room connection via `useLiveKit` hook
   - Real-time transcript display
   - Segment-based UI with topic grouping
   - AI nudges panel (key proposals, risks, actions)
   - One-click redeploy button

### Audio Processing Pipeline (M4 Optimized)

```
LiveKit AudioStream (48kHz int16)
  ↓
RESAMPLE TO 16kHz (3x faster VAD processing)
  └─ scipy.signal.resample_poly (efficient 3:1 decimation)
  ↓
Audio Preprocessing (runs at 16kHz)
  ├─ High-pass filter (100Hz) - removes rumble/DC offset
  ├─ Peak normalization (-3dB, optional) - consistent levels
  └─ Quality validation - prevents hallucinations
  ↓
Silero VAD (M4-optimized settings for continuous speech)
  ├─ activation_threshold: 0.35 (catches softer speech)
  ├─ min_speech_duration: 0.25s (faster detection)
  ├─ min_silence_duration: 1.2s (tolerates natural pauses)
  └─ Speech segment detection
  ↓
MLX Whisper (whisper-large-v3-turbo) - NON-BLOCKING
  ├─ asyncio.create_task() for parallel transcription
  ├─ VAD continues listening while Whisper processes
  └─ Transcribe audio segment (M4 Neural Engine)
  ↓
LiveKit Data Channel → Frontend Display
  ↓
LLM Classifier (SEPARATE PROCESS - CPU isolated)
  ├─ MLX-LM Qwen2.5-1.5B classifier
  ├─ Topic segmentation (NEW_TOPIC/CONTINUE/NOISE)
  └─ Triggers nudge generation on segment completion
  ↓
GPT-4.1 Nudge Generation (via LiteLLM)
  └─ Categorizes into: key proposals, delivery risks, action items
```

## Manual Setup (Alternative to setup.sh)

### 1. Install Dependencies

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Node.js and LiveKit
brew install node
brew install livekit

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# Create .env file (if not exists)
cp .env.example .env
# Edit .env with your API keys
```

**Required environment variables** (`backend/.env`):
```env
# LiveKit Configuration (local dev defaults)
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# OpenAI API Key (required for nudges)
OPENAI_API_KEY=your-openai-api-key

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
```

### 3. Frontend Setup

```bash
cd frontend
npm install

# Create .env.local (if not exists)
cat > .env.local << 'EOF'
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/transcripts
NEXT_PUBLIC_API_URL=http://localhost:8000/api
EOF
```

## Running the Application

### Automated Start (Recommended)

```bash
# Start all services with one command
./start.sh

# Check service status
./status.sh

# Stop all services
./stop.sh
```

### Manual Start (Alternative)

Start services in separate terminals:

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

**Access the app:** http://localhost:3000

## Usage

1. **Open** http://localhost:3000 in your browser
2. **Click "Mic On"** to enable microphone
3. **Start speaking** - Transcripts appear in real-time
4. **View segments** - Topics automatically grouped
5. **See nudges** - AI insights on completed segments:
   - 🎯 Key Proposals
   - ⚠️ Delivery Risks
   - ✅ Action Items
6. **Export** - Download meeting data as JSON

## Troubleshooting

### No Transcripts Appearing

**Solution 1: Check microphone permissions**
1. Click lock icon in browser address bar
2. Ensure microphone is allowed
3. Reload page if needed

**Solution 2: Verify microphone selection**
1. Open **System Settings → Sound → Input**
2. Confirm correct microphone selected (not BlackHole/virtual device)
3. Verify input level meter moves when speaking

**Solution 3: Use the Redeploy button**
1. Click **"Redeploy"** button in top-right corner
2. Confirm the dialog (system will restart in ~20 seconds)
3. Page will automatically reload when ready

**Solution 4: Manual restart**
```bash
./stop.sh
./start.sh
```

**Solution 5: Check logs**
```bash
# View logs for debugging
tail -f /tmp/agent.log      # Agent logs
tail -f /tmp/fastapi.log    # API server logs
tail -f /tmp/livekit.log    # LiveKit server logs
tail -f /tmp/nextjs.log     # Frontend logs
```

Look for:
- "Audio is near-silent" warnings (microphone issue)
- "Speech started" / "Speech ended" messages (VAD working)
- "registered worker" message (agent connected)

### MLX Whisper Not Loading

**Verify Apple Silicon:**
```bash
uname -m  # Should show "arm64"
```

**Reinstall MLX Whisper:**
```bash
cd backend
source .venv/bin/activate
uv pip install --force-reinstall mlx-whisper
```

### LiveKit Connection Failed

**Check LiveKit server running:**
```bash
ps aux | grep livekit-server
# Should show: livekit-server --dev
```

**Restart LiveKit:**
```bash
pkill livekit-server
livekit-server --dev
```

### Agent Not Joining Room

**Verify all services running:**
```bash
./status.sh
```

**Check agent logs:**
```bash
tail -f /tmp/agent.log
# Look for "registered worker" message
```

### WebSocket Disconnected

**Ensure FastAPI running:**
```bash
lsof -ti:8000
# Should return a PID
```

### Speech Loss / Fragmented Transcripts

This is already fixed in M4 optimizations:
- `min_silence_duration: 1.2s` (tolerates breathing pauses)
- `activation_threshold: 0.35` (catches softer speech)
- Non-blocking transcription prevents speech loss

### Whisper Hallucinations

If seeing random/repeated text, audio validation is working. Causes:
- Audio too quiet (below -55dB)
- Segment too short (<0.3s)
- Corrupted/clipped audio

**Solution:** Adjust microphone gain in System Settings → Sound

## API Endpoints

### REST API

- `POST /api/livekit/token` - Generate access token and dispatch agent
- `POST /api/nudge` - Generate AI insights from transcripts
  ```json
  {
    "transcripts": ["text1", "text2"],
    "segment_id": "optional-segment-id",
    "topic": "optional-topic-context"
  }
  ```
- `POST /api/redeploy` - Restart all system services
- `GET /api/meetings` - List all meeting sessions
- `GET /api/meetings/{id}` - Get meeting details

### WebSocket

- `ws://localhost:8000/ws/transcripts` - Real-time transcript stream (optional)

## Key Features

### M4-Specific Optimizations
- **16kHz resampling before VAD** (3x faster processing)
- **MLX Neural Engine utilization** via environment flags
- **Multiprocessing for LLM** (CPU isolation from audio pipeline)
- **Relaxed VAD thresholds** for continuous speech
- **Lower silence threshold** (-55dB) for quieter speech

### Audio Quality Validation
Prevents Whisper hallucinations by rejecting:
- Empty/corrupted audio
- Near-silent audio (below -55dB RMS)
- Stuck buffers (constant signal)
- Excessive clipping (>3%)
- Too-short segments (<0.3s)

### Non-Blocking Architecture
- **VAD continues listening** during transcription
- **LLM classifier** runs in separate process
- **Nudge generation** triggered on segment completion
- **Zero speech loss** due to background processing

## Performance Metrics

- **Transcription latency:** ~1-2s (MLX Whisper Turbo on M4)
- **Audio preprocessing:** <5ms per frame (real-time at 16kHz)
- **VAD detection:** Real-time, no "slower than realtime" warnings
- **LLM classification:** Non-blocking (separate process)
- **Model loading:** ~5s on first room join (cached thereafter)

## Technology Stack

### Backend
- **LiveKit Agents SDK** - Real-time audio processing
- **Silero VAD** - Voice activity detection
- **MLX Whisper** - Apple Silicon optimized transcription (whisper-large-v3-turbo)
- **MLX-LM** - Topic classification (Qwen2.5-1.5B)
- **FastAPI** - High-performance async API
- **GPT-4.1** - AI-powered nudges (via LiteLLM proxy)
- **SciPy** - Audio signal processing
- **Python 3.10+** with `uv` package manager

### Frontend
- **Next.js 16** - React framework with App Router
- **React 19** - Latest React features
- **TypeScript** - Type-safe development
- **Tailwind CSS v4** - Modern styling with custom theme
- **LiveKit Client SDK** - Room connection and data channels
- **Framer Motion** - Smooth UI animations
- **Recharts** - Data visualization
- **Lucide React** - Icon system

## Project Structure

```
DTS-Voice/
├── backend/
│   ├── agent/
│   │   ├── production_agent.py      # Main LiveKit agent (M4 optimized)
│   │   ├── audio_preprocessing.py   # Audio filter/normalize/validate
│   │   ├── transcription.py         # MLX Whisper wrapper
│   │   ├── llm_classifier.py        # MLX-LM topic classification
│   │   ├── llm_classifier_worker.py # Separate process worker
│   │   ├── segment_manager.py       # Real-time topic segmentation
│   │   └── config.py                # Environment configuration
│   └── api/
│       ├── server.py                # FastAPI server
│       ├── llm_service.py           # GPT-4.1 nudge generation
│       └── storage.py               # JSON file operations
├── frontend/
│   ├── app/
│   │   └── page.tsx                 # Main page with transcript/nudges
│   ├── components/
│   │   ├── LiveTranscript.tsx       # Real-time transcript display
│   │   ├── NudgesPanel.tsx          # AI nudges sidebar
│   │   └── MeetingControls.tsx      # Mic/export controls
│   ├── hooks/
│   │   └── useLiveKit.ts            # LiveKit connection hook
│   └── lib/
│       └── types.ts                 # TypeScript interfaces
├── scripts/
│   └── setup.sh                     # One-command installation
├── data/                            # Meeting data & logs (gitignored)
├── start.sh                         # Start all services
├── stop.sh                          # Stop all services
├── status.sh                        # Check service health
└── README.md                        # This file
```

## Scripts

### Setup & Management

- **`./scripts/setup.sh`** - One-command installation (installs uv, Node.js, LiveKit, dependencies)
- **`./start.sh`** - Start all services (LiveKit, FastAPI, Agent, Frontend)
- **`./stop.sh`** - Clean shutdown of all services
- **`./status.sh`** - Check service health and ports

### Logs

All logs stored in `/tmp/`:
- `/tmp/livekit.log` - LiveKit server
- `/tmp/fastapi.log` - FastAPI backend
- `/tmp/agent.log` - Production agent
- `/tmp/nextjs.log` - Next.js frontend
- `/tmp/redeploy.log` - Redeploy operations

Debug logs (in `data/`):
- `data/llm_classifier_debug.log` - LLM prompts and responses
- `data/segmentation_debug.log` - Segment state changes

## Configuration Files

### Backend Environment (`backend/.env`)
- LiveKit credentials (local dev uses `devkey/secret`)
- OpenAI API key for nudges
- M4 optimization flags
- Audio preprocessing settings
- Segmentation parameters

### Frontend Environment (`frontend/.env.local`)
- LiveKit WebSocket URL
- FastAPI API URL
- WebSocket URL (optional)

## Development

### Backend Development

```bash
cd backend
source .venv/bin/activate

# Run agent in dev mode
python -m agent.production_agent dev

# Run API server with auto-reload
python -m uvicorn api.server:app --reload
```

### Frontend Development

```bash
cd frontend

# Run dev server
npm run dev

# Build for production
npm run build

# Lint code
npm run lint
```

## Production Deployment

For production deployment:

1. **Use LiveKit Cloud** instead of local server
2. **Update environment variables** with production credentials
3. **Build frontend:** `cd frontend && npm run build`
4. **Run frontend:** `npm start`
5. **Use process manager** (PM2, systemd) for backend services
6. **Enable HTTPS** for secure WebSocket connections

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **LiveKit** - Real-time communication platform
- **OpenAI** - GPT-4.1 language model
- **Apple** - MLX framework for Apple Silicon
- **Anthropic** - Claude AI development assistance
- **Hugging Face** - MLX-LM and model hosting

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check logs in `/tmp/` for debugging
- Use the Redeploy button for quick restarts

---

**Built with ❤️ for real-time meeting intelligence**
