# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time meeting transcription system with AI-powered insights. Audio flows from LiveKit → Audio Preprocessing → Silero VAD → MLX Whisper → FastAPI WebSocket → Next.js UI. OpenAI GPT-4 classifies transcripts into nudges (proposals, risks, actions).

**Critical**: Requires Apple Silicon for MLX Whisper. Must use `uv` for Python dependency management (mandatory).

## Development Commands

### Backend (Python)

```bash
# Setup
cd backend
uv venv
source .venv/bin/activate
uv pip install -e .

# Run FastAPI server (port 8000)
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# Run LiveKit production agent (recommended)
python -m agent.production_agent dev

# Or run basic agent (legacy)
python -m agent.main

# Both API server and agent required for full functionality
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
npm run build
npm run lint
```

### Quick Start

```bash
# Start LiveKit server
livekit-server --dev

# Start backend services (in separate terminals)
cd backend && source .venv/bin/activate
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
python -m agent.production_agent dev

# Start frontend
cd frontend && npm run dev
```

## Architecture

### Three-Service Model

1. **LiveKit Production Agent** (`backend/agent/production_agent.py`)
   - Joins LiveKit rooms via job dispatch API
   - Applies real-time audio preprocessing (high-pass filter, normalization)
   - Uses Silero VAD for speech detection with production settings
   - Transcribes audio using MLX Whisper (whisper-large-v3-turbo)
   - Validates audio quality to prevent hallucinations
   - Sends transcripts to frontend via LiveKit data channel

2. **FastAPI Server** (`backend/api/server.py`)
   - Generates LiveKit access tokens for participants
   - Dispatches agents to rooms automatically
   - WebSocket at `/ws/transcripts` for real-time updates (optional)
   - REST endpoint `/api/nudge` for LLM classification
   - CORS configured for localhost:3000

3. **Next.js Frontend** (`frontend/app/page.tsx`)
   - `useLiveKit` hook manages LiveKit room connection
   - Receives transcripts via LiveKit data channel
   - Auto-requests nudges every 5 final transcripts
   - Dark theme via Tailwind v4 (`@import "tailwindcss"`)

### Production Agent Pipeline

**Audio Processing Flow**:
```
LiveKit AudioStream (48kHz int16)
  → Audio Preprocessing Module
      ├─ High-pass filter (80Hz) - removes rumble/DC offset
      ├─ Peak normalization (-3dB) - consistent levels
      └─ Quality validation - prevents hallucinations
  → Silero VAD (production settings)
      ├─ activation_threshold: 0.4
      ├─ min_speech_duration: 0.3s
      ├─ min_silence_duration: 0.6s
      └─ Speech segment detection
  → MLX Whisper (whisper-large-v3-turbo)
      ├─ Resample 48kHz → 16kHz
      ├─ Transcribe audio segment
      └─ Return text + confidence
  → LiveKit Data Channel
      └─ Send to frontend participants
```

**Audio Validation Checks** (prevents Whisper hallucinations):
- Empty audio detection
- NaN/Inf value detection
- Silence threshold (-55dB RMS)
- Stuck buffer detection
- Clipping detection (>3%)
- Minimum duration (0.3s)

### Data Flow Pattern

**Audio → Transcript**:
```
LiveKit Room → production_agent.py processes audio
  → Audio preprocessing (filter + normalize)
  → VAD detects speech boundaries
  → transcription.transcribe_audio() [MLX Whisper]
  → Send via LiveKit data channel
  → Frontend receives and displays
```

**Transcript → Nudges**:
```
Frontend accumulates transcripts → POST /api/nudge
  → llm_service.classify_transcripts() [OpenAI GPT-4]
  → Returns structured JSON → Frontend displays
```

### Key Patterns

- **Production VAD Settings**: Balanced sensitivity (0.4 threshold) reduces false positives
- **Audio Preprocessing**: Lightweight real-time processing prevents corruption
- **Quality Validation**: Multi-check system prevents Whisper hallucinations on bad audio
- **Agent Dispatch**: Automatic dispatch via API server when participants join
- **Data Channel Communication**: Direct LiveKit data channel (no WebSocket dependency)
- **Functional Style**: Simple functions preferred over classes

## Module Responsibilities

### `backend/agent/`

- **production_agent.py**: Production LiveKit agent with audio preprocessing and VAD
- **main.py**: Legacy basic agent (simpler, less robust)
- **audio_preprocessing.py**: Real-time audio processing (filter, normalize, validate)
- **transcription.py**: MLX Whisper wrapper - `transcribe_audio()` returns dict with text/confidence
- **config.py**: Environment variables and production settings

### `backend/api/`

- **server.py**: FastAPI app with token generation, agent dispatch, and REST endpoints
- **llm_service.py**: OpenAI integration with prompt template for classification
- **storage.py**: JSON file operations for meeting data

### `frontend/`

- **app/page.tsx**: Main page with transcript/nudges panels
- **components/**: `LiveTranscript`, `NudgesPanel`, `MeetingControls`
- **hooks/useLiveKit.ts**: LiveKit room connection and microphone management
- **lib/types.ts**: TypeScript interfaces for Transcript, Nudge, etc.

## Configuration

### Required Environment Variables

**backend/.env**:
```env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
OPENAI_API_KEY=<your-openai-key>

# Optional: Audio preprocessing settings
ENABLE_HIGHPASS_FILTER=true
ENABLE_NORMALIZATION=true
HIGHPASS_CUTOFF_HZ=80
NORMALIZATION_TARGET_DB=-3.0

# Optional: Audio validation settings
ENABLE_AUDIO_VALIDATION=true
AUDIO_VALIDATION_RMS_THRESHOLD_DB=-55.0
```

**frontend/.env.local**:
```env
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/transcripts
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### LiveKit Server

Must run separately:
```bash
livekit-server --dev  # Local dev mode (uses devkey/secret)
# Or use LiveKit Cloud credentials
```

### Whisper Model Configuration

In `backend/agent/config.py`:
```python
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"  # Recommended
# Alternative: "mlx-community/whisper-large-v3" (slower, more accurate)
```

## Common Issues

### MLX Whisper Installation
Only works on Apple Silicon. If `uv pip install mlx-whisper` fails, verify `uname -m` shows `arm64`.

### Audio is Silent / No Transcripts
1. Check browser microphone permissions (lock icon in address bar)
2. Verify correct microphone selected in System Settings → Sound → Input
3. Check agent logs for "Audio is near-silent" warnings
4. Try adjusting VAD settings in `production_agent.py` if too sensitive/insensitive

### WebSocket Connection Errors
Frontend shows "Disconnected" if API server not running on port 8000. Start with:
```bash
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

### Agent Not Joining Room
1. Verify LiveKit server running (port 7880)
2. Check `.env` credentials match (devkey/secret for local dev)
3. Ensure API server is running to dispatch agents
4. Check agent logs for "registered worker" message

### Whisper Hallucinations
If seeing random/repeated text, the audio validation is working. This happens when:
- Audio is too quiet (below -55dB)
- Audio segment too short (<0.3s)
- Audio is corrupted or clipped

Solution: Adjust thresholds in `config.py` or improve microphone gain.

### Tailwind CSS Errors
Uses Tailwind v4 syntax: `@import "tailwindcss"` not `@tailwind base/components/utilities`. Custom colors via `@theme` block.

## Code Style

- **Functional over OOP**: Use functions, avoid classes unless necessary
- **Type hints**: Python functions have type hints, frontend uses TypeScript
- **Async/await**: All agent and API code is async
- **Simple imports**: Relative imports within modules (e.g., `from . import config`)
- **Error handling**: Log errors visibly, don't silently fail

## Production Settings

### VAD Configuration (production_agent.py)
```python
activation_threshold=0.4       # Balanced sensitivity
min_speech_duration=0.3        # Reduces false positives
min_silence_duration=0.6       # Reliable speech end detection
prefix_padding_duration=0.3    # Captures word starts
```

### Audio Validation (config.py)
```python
AUDIO_VALIDATION_RMS_THRESHOLD_DB = -55.0  # Silence threshold
```

### Audio Preprocessing
- High-pass filter: 80Hz (removes rumble, AC hum)
- Peak normalization: -3dB (prevents clipping)
- Sample rate: 48kHz → 16kHz resampling for Whisper

## Dependencies Management

**Python**: Must use `uv` (not pip). All deps in `backend/pyproject.toml`.

**Node**: Standard npm. Key deps: `livekit-client`, `recharts`, `lucide-react`.

## Performance Notes

- Transcription latency: ~1-2s (MLX Whisper Turbo on M4 Apple Silicon)
- Audio preprocessing: <5ms per frame (lightweight, real-time)
- VAD detection: Real-time with optimized settings
- Model loading: ~5s on first room join (cached thereafter)

## UI Theme

Dark theme colors (Tailwind v4 custom theme):
- Background: `#0f1419`
- Panel: `#1a1f2e`
- Card: `#252b3d`
- Accent green: `#4ade80`
- Accent amber: `#fbbf24`
