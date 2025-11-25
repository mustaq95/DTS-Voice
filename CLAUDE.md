# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time meeting transcription system with AI-powered insights. Audio flows from LiveKit → Silero VAD → MLX Whisper → FastAPI WebSocket → Next.js UI. OpenAI GPT-4 classifies transcripts into nudges (proposals, risks, actions).

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

# Run LiveKit agent
python -m agent.main

# Both required for full functionality
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
./scripts/setup.sh              # First-time setup
./scripts/start-backend.sh      # Start API + Agent
./scripts/start-frontend.sh     # Start Next.js
```

## Architecture

### Three-Service Model

1. **LiveKit Agent** (`backend/agent/main.py`)
   - Joins LiveKit room as participant
   - Receives audio via `rtc.AudioStream`
   - Uses `@server.rtc_session()` decorator for entry point
   - Global `meeting_sessions` dict stores active meetings

2. **FastAPI Server** (`backend/api/server.py`)
   - WebSocket at `/ws/transcripts` broadcasts transcripts
   - REST endpoint `/api/nudge` for LLM classification
   - `ConnectionManager` class handles WebSocket clients
   - CORS configured for localhost:3000

3. **Next.js Frontend** (`frontend/app/page.tsx`)
   - `useTranscription` hook manages WebSocket connection
   - Auto-requests nudges every 5 final transcripts
   - Dark theme via Tailwind v4 (`@import "tailwindcss"`)

### Data Flow Pattern

**Audio → Transcript**:
```
LiveKit Room → agent.main.handle_participant_audio()
  → VAD detects speech end → audio_buffer accumulated
  → transcription.transcribe_audio() [MLX Whisper]
  → storage.add_transcript() → JSON file
  → WebSocket broadcast → Frontend
```

**Transcript → Nudges**:
```
Frontend accumulates transcripts → POST /api/nudge
  → llm_service.classify_transcripts() [OpenAI GPT-4]
  → Returns structured JSON → Frontend displays
```

### Key Patterns

- **Agent State**: Global `meeting_sessions` dict in `agent/main.py` stores per-room sessions
- **Storage**: Simple JSON files in `data/meetings/{uuid}.json` - no database
- **WebSocket Broadcast**: `ConnectionManager` in `server.py` broadcasts to all connected clients
- **Functional Style**: No classes except where required (Agent, ConnectionManager) - simple functions preferred

## Module Responsibilities

### `backend/agent/`

- **main.py**: LiveKit agent entry point using `AgentServer` and `@server.rtc_session()` decorator
- **transcription.py**: MLX Whisper wrapper - `transcribe_audio()` returns dict with text/confidence
- **vad_handler.py**: Silero VAD event handlers (currently simplified, VAD loaded but not fully integrated)
- **config.py**: Environment variables via `python-dotenv`

### `backend/api/`

- **server.py**: FastAPI app with WebSocket `/ws/transcripts` and REST endpoints
- **llm_service.py**: OpenAI integration with prompt template for classification
- **storage.py**: JSON file operations - `create_meeting_session()`, `add_transcript()`, `save_meeting_session()`

### `frontend/`

- **app/page.tsx**: Main page with transcript/nudges panels
- **components/**: `LiveTranscript`, `NudgesPanel`, `MeetingControls`
- **hooks/useTranscription.ts**: WebSocket client managing connection and state
- **lib/types.ts**: TypeScript interfaces for Transcript, Nudge, etc.

## Configuration

### Required Environment Variables

**backend/.env**:
```env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=<livekit-key>
LIVEKIT_API_SECRET=<livekit-secret>
OPENAI_API_KEY=<openai-key>
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
livekit-server --dev  # Local, or use LiveKit Cloud credentials
```

## Common Issues

### MLX Whisper Installation
Only works on Apple Silicon. If `uv pip install mlx-whisper` fails, verify `uname -m` shows `arm64`.

### WebSocket Connection Errors
Frontend shows "WebSocket disconnected" if FastAPI server not running. Start with `python -m uvicorn api.server:app --reload`.

### Agent Not Transcribing
1. Verify LiveKit server running (port 7880)
2. Check `.env` credentials are correct
3. Ensure agent connected: check logs for "Agent joining room"
4. Audio frames need proper conversion in `main.py:process_speech_segment()` - currently simplified

### Tailwind CSS Errors
Uses Tailwind v4 syntax: `@import "tailwindcss"` not `@tailwind base/components/utilities`. Custom colors via `@theme` block.

## Code Style

- **Functional over OOP**: Use functions, avoid classes unless necessary
- **Type hints**: Python functions have type hints, frontend uses TypeScript
- **Async/await**: All agent and API code is async
- **Simple imports**: Relative imports within modules (e.g., `from . import config`)

## Data Schemas

### Meeting Session JSON
```json
{
  "meeting_id": "uuid",
  "room_name": "voice-fest",
  "started_at": "ISO8601",
  "transcripts": [
    {"timestamp": "HH:MM:SS", "speaker": "string", "text": "string", "is_final": bool}
  ],
  "nudges": [
    {"type": "key_proposal|delivery_risk|action_item", "title": "string", "quote": "string", "confidence": float}
  ]
}
```

### WebSocket Message Format
```json
{
  "type": "transcript",
  "data": {"timestamp": "10:30:15", "speaker": "Chair", "text": "...", "is_final": true}
}
```

## Dependencies Management

**Python**: Must use `uv` (not pip). All deps in `backend/pyproject.toml`.

**Node**: Standard npm. Key deps: `livekit-client`, `recharts`, `lucide-react`.

## Testing Notes

No tests currently implemented. When adding:
- Backend: Use `pytest` with fixtures for LiveKit/WebSocket mocking
- Frontend: Use React Testing Library for components, Playwright for E2E
- Integration: Test full flow with mock LiveKit room

## Performance Notes

- Transcription latency: ~1-2s (MLX Whisper Tiny on Apple Silicon)
- VAD currently simplified - full integration would use Silero VAD events properly
- WebSocket has no reconnection backoff - consider adding for production
- No rate limiting on `/api/nudge` endpoint

## UI Theme

Dark theme colors (Tailwind v4 custom theme):
- Background: `#0f1419`
- Panel: `#1a1f2e`
- Card: `#252b3d`
- Accent green: `#4ade80`
- Accent amber: `#fbbf24`
