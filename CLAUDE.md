# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time meeting transcription system with AI-powered topic segmentation. Audio flows from LiveKit → Audio Preprocessing (16kHz) → Silero VAD → MLX Whisper → LiveKit Data Channel → Next.js UI. MLX-LM classifier (running in separate process) segments transcripts by topic.

**Critical**: Requires Apple Silicon for MLX Whisper/MLX-LM. Must use `uv` for Python dependency management (mandatory). Optimized for Apple M4 chip.

## Development Commands

### Quick Start (Recommended)

```bash
# One-command startup (all services)
./start.sh

# Check service status
./status.sh

# Stop all services
./stop.sh
```

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

### Manual Start (if not using scripts)

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
   - **Resamples audio 48kHz → 16kHz BEFORE VAD** (3x performance boost)
   - Applies real-time audio preprocessing (high-pass filter, normalization)
   - Uses Silero VAD with M4-optimized settings (continuous speech capture)
   - Transcribes audio using MLX Whisper (whisper-large-v3-turbo)
   - **Non-blocking transcription** - VAD continues listening during transcription
   - Validates audio quality to prevent hallucinations
   - Sends transcripts to frontend via LiveKit data channel
   - Topic segmentation via LLM classifier (separate process)

2. **FastAPI Server** (`backend/api/server.py`)
   - Generates LiveKit access tokens for participants
   - Dispatches agents to rooms automatically
   - WebSocket at `/ws/transcripts` for real-time updates (optional)
   - REST endpoint `/api/nudge` for LLM classification
   - CORS configured for localhost:3000

3. **Next.js Frontend** (`frontend/app/page.tsx`)
   - `useLiveKit` hook manages LiveKit room connection
   - Receives transcripts via LiveKit data channel
   - Displays live transcripts and topic segments
   - Dark theme via Tailwind v4 (`@import "tailwindcss"`)

### Production Agent Pipeline

**Audio Processing Flow** (M4 Optimized):
```
LiveKit AudioStream (48kHz int16)
  → RESAMPLE TO 16kHz (NEW: 3x faster VAD processing)
      └─ scipy.signal.resample_poly (efficient 3:1 decimation)
  → Audio Preprocessing Module (runs at 16kHz - 3x faster)
      ├─ High-pass filter (100Hz) - removes rumble/DC offset
      ├─ Peak normalization (-3dB, optional) - consistent levels
      └─ Quality validation - prevents hallucinations
  → Silero VAD (M4-optimized settings for continuous speech)
      ├─ activation_threshold: 0.35 (catches softer speech)
      ├─ min_speech_duration: 0.25s (faster detection)
      ├─ min_silence_duration: 1.2s (tolerates natural pauses)
      ├─ prefix_padding_duration: 0.4s (captures word starts)
      └─ Speech segment detection
  → MLX Whisper (whisper-large-v3-turbo) - NON-BLOCKING
      ├─ asyncio.create_task() for parallel transcription
      ├─ VAD continues listening while Whisper processes
      ├─ Transcribe audio segment (M4 Neural Engine)
      └─ Return text + confidence
  → LiveKit Data Channel
      └─ Send to frontend participants
  → LLM Classifier (SEPARATE PROCESS - CPU isolated)
      ├─ MLX-LM Qwen2.5-1.5B classifier
      ├─ Queue-based async communication
      └─ Topic segmentation in background
```

**Audio Validation Checks** (prevents Whisper hallucinations):
- Empty audio detection
- NaN/Inf value detection
- Silence threshold (-55dB RMS, M4-optimized)
- Stuck buffer detection
- Clipping detection (>3%)
- Minimum duration (0.3s)

### Data Flow Pattern

**Audio → Transcript** (Non-Blocking):
```
LiveKit Room → production_agent.py processes audio
  → Resample 48kHz → 16kHz (BEFORE VAD)
  → Audio preprocessing at 16kHz (3x faster)
  → VAD detects speech boundaries
  → asyncio.create_task(transcribe_and_publish) [NON-BLOCKING]
      ├─ VAD continues listening (no blocking)
      └─ transcription.transcribe_audio() [MLX Whisper]
  → Send via LiveKit data channel
  → Frontend receives and displays
```

**Transcript → Topic Segmentation** (Separate Process):
```
production_agent.py → segment_manager.add_transcript() [NON-BLOCKING]
  → LLM Classifier Worker Process (multiprocessing.Queue)
      ├─ MLX-LM Qwen2.5-1.5B classification
      ├─ CPU isolated from main agent
      └─ Returns topic classification
  → Update segment state
  → Save to JSON and publish
```

### Key Patterns

- **M4-Optimized VAD**: Lower threshold (0.35), longer silence tolerance (1.2s) for continuous speech
- **16kHz Preprocessing**: Resample BEFORE VAD (3x faster processing)
- **Non-Blocking Transcription**: asyncio.create_task() prevents speech loss
- **CPU Isolation**: LLM classifier in separate process (no VAD blocking)
- **Audio Preprocessing**: Lightweight real-time processing at 16kHz
- **Quality Validation**: Multi-check system prevents Whisper hallucinations
- **Agent Dispatch**: Automatic dispatch via API server when participants join
- **Data Channel Communication**: Direct LiveKit data channel (no WebSocket dependency)
- **Functional Style**: Simple functions preferred over classes

## Module Responsibilities

### `backend/agent/`

- **production_agent.py**: Production LiveKit agent with M4 optimizations, non-blocking transcription, multiprocessing setup
- **audio_preprocessing.py**: Real-time audio processing (filter, normalize, validate) - runs at 16kHz
- **transcription.py**: MLX Whisper wrapper - `transcribe_audio()` returns dict with text/confidence
- **llm_classifier_worker.py**: **NEW** - Separate process worker for LLM classification (CPU isolated)
- **segment_manager.py**: **NEW** - Real-time topic segmentation with async processing
- **llm_classifier.py**: MLX-LM classification logic (used by worker)
- **config.py**: Environment variables and M4-optimized settings
- **main.py**: Legacy basic agent (deprecated, use production_agent.py)

### `backend/api/`

- **server.py**: FastAPI app with token generation, agent dispatch, and REST endpoints
- **llm_service.py**: OpenAI integration (optional, for nudges)
- **storage.py**: JSON file operations for meeting data

### `frontend/`

- **app/page.tsx**: Main page with transcript/nudges panels
- **components/**: `LiveTranscript`, `NudgesPanel`, `MeetingControls`
- **hooks/useLiveKit.ts**: LiveKit room connection and microphone management
- **lib/types.ts**: TypeScript interfaces for Transcript, Nudge, Segment, etc.

### Shell Scripts

- **start.sh**: **NEW** - Launch all services (LiveKit, FastAPI, Agent, Frontend)
- **stop.sh**: **NEW** - Clean shutdown of all services
- **status.sh**: **NEW** - Check service health and ports

## Configuration

### Required Environment Variables

**backend/.env**:
```env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
OPENAI_API_KEY=<your-openai-key>

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
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"  # M4-optimized (fast & accurate)
```

## Common Issues

### MLX Whisper Installation
Only works on Apple Silicon. If `uv pip install mlx-whisper` fails, verify `uname -m` shows `arm64`.

### Audio is Silent / No Transcripts
1. Check browser microphone permissions (lock icon in address bar)
2. Verify correct microphone selected in System Settings → Sound → Input
3. Check agent logs for "Audio is near-silent" warnings
4. Verify VAD settings in `production_agent.py` (should be 0.35 threshold for M4)

### Speech Loss / Fragmented Transcripts
**Symptom**: Seeing fragments like "be watching?", "news is...", "a point." instead of continuous sentences.

**Cause**: VAD settings too strict, splitting speech on natural pauses.

**Solution**: Already fixed in M4 optimizations:
- `min_silence_duration: 1.2s` (tolerates breathing/thinking pauses)
- `activation_threshold: 0.35` (catches softer speech)

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

### "inference is slower than realtime" Warnings
**Cause**: VAD processing slower than audio input (usually CPU bottleneck).

**Solution**: Already fixed via:
1. 16kHz resampling BEFORE VAD (3x faster)
2. LLM classifier in separate process (CPU isolation)

If still seeing warnings, check system CPU load.

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
- **Non-blocking**: Use `asyncio.create_task()` for background operations
- **Simple imports**: Relative imports within modules (e.g., `from . import config`)
- **Error handling**: Log errors visibly, don't silently fail

## Production Settings (M4 Optimized)

### VAD Configuration (production_agent.py)
```python
activation_threshold=0.35      # M4 OPTIMIZED: Lower threshold for softer speech
min_speech_duration=0.25       # M4 OPTIMIZED: Faster detection start
min_silence_duration=1.2       # M4 OPTIMIZED: Longer tolerance for natural pauses
prefix_padding_duration=0.4    # M4 OPTIMIZED: More padding to capture word starts
```

**Impact**: Prevents sentence fragmentation, enables continuous speech capture.

### Audio Validation (config.py)
```python
AUDIO_VALIDATION_RMS_THRESHOLD_DB = -55.0  # M4 OPTIMIZED: More permissive
```

### Audio Preprocessing
- High-pass filter: 100Hz (removes rumble, AC hum)
- Peak normalization: -3dB (optional, disabled by default)
- **Sample rate: 48kHz → 16kHz resampling BEFORE VAD** (3x performance boost)

## Dependencies Management

**Python**: Must use `uv` (not pip). All deps in `backend/pyproject.toml`.

**Node**: Standard npm. Key deps: `livekit-client`, `recharts`, `lucide-react`.

## Performance Notes (M4 Apple Silicon)

- **Transcription latency**: ~1-2s (MLX Whisper Turbo on M4 Neural Engine)
- **Audio preprocessing**: <5ms per frame (lightweight, real-time at 16kHz)
- **VAD detection**: Real-time, no "slower than realtime" warnings
- **LLM classification**: Runs in separate process (CPU isolated, non-blocking)
- **Model loading**: ~5s on first room join (cached thereafter)
- **Zero speech loss**: Non-blocking transcription + optimized VAD

### M4-Specific Optimizations
- MLX Neural Engine utilization via environment flags
- 16kHz audio preprocessing (3x faster than 48kHz)
- Multiprocessing for LLM (CPU isolation)
- Relaxed VAD thresholds for continuous speech
- Lower silence threshold (-55dB) for quieter speech

## UI Theme

Dark theme colors (Tailwind v4 custom theme):
- Background: `#0f1419`
- Panel: `#1a1f2e`
- Card: `#252b3d`
- Accent green: `#4ade80`
- Accent amber: `#fbbf24`

## Troubleshooting Tips

### Check Service Health
```bash
./status.sh  # Shows all running services and ports
```

### View Logs
```bash
tail -f /tmp/agent.log      # Agent logs
tail -f /tmp/fastapi.log    # API server logs
tail -f /tmp/nextjs.log     # Frontend logs
tail -f /tmp/livekit.log    # LiveKit server logs
```

### Debug LLM Classifier
```bash
tail -f data/llm_classifier_debug.log  # LLM prompts and responses
tail -f data/segmentation_debug.log    # Segment state changes
```

### Verify M4 Optimizations Active
```bash
# Check agent log for M4-optimized values
grep -E "activation_threshold|min_silence" /tmp/agent.log

# Should see:
# activation_threshold=0.35
# min_silence_duration=1.2
```
