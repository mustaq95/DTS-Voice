# LiveKit Meeting Intelligence System

Real-time meeting transcription and intelligence system powered by LiveKit, MLX Whisper, and OpenAI.

## Overview

This project provides:
- **Real-time transcription** using MLX Whisper (optimized for Apple Silicon)
- **Voice Activity Detection** using Silero VAD through LiveKit
- **LLM-powered insights** via OpenAI GPT-4 for extracting key proposals, risks, and action items
- **Modern web interface** built with Next.js showing live transcripts and AI-generated nudges

## Project Structure

```
livekit-whisper/
├── backend/                    # Python backend
│   ├── agent/                 # LiveKit agent
│   │   ├── main.py           # Agent entry point
│   │   ├── transcription.py  # MLX Whisper integration
│   │   ├── vad_handler.py    # Silero VAD handling
│   │   └── config.py         # Configuration
│   └── api/                   # FastAPI server
│       ├── server.py         # WebSocket & REST API
│       ├── llm_service.py    # OpenAI integration
│       └── storage.py        # JSON file storage
├── frontend/                  # Next.js frontend
│   ├── app/                  # Next.js app directory
│   ├── components/           # React components
│   ├── hooks/               # React hooks
│   └── lib/                 # TypeScript types
├── data/                     # Meeting data storage
│   ├── transcripts/         # Transcript JSON files
│   └── meetings/            # Meeting session data
├── docs/                    # Documentation
└── scripts/                 # Utility scripts
```

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
source .venv/bin/activate  # On macOS/Linux
uv pip install -e .

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Required environment variables in `.env`:
```env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret
OPENAI_API_KEY=your-openai-api-key
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

**Option A: Local Development**
```bash
# Install LiveKit server
brew install livekit

# Start server
livekit-server --dev
```

**Option B: LiveKit Cloud**
Sign up at [livekit.io](https://livekit.io) and use your cloud credentials.

## Running the Application

### Start Backend Services

**Terminal 1: FastAPI Server**
```bash
cd backend
source .venv/bin/activate
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2: LiveKit Agent**
```bash
cd backend
source .venv/bin/activate
python -m agent.main
```

### Start Frontend

**Terminal 3: Next.js**
```bash
cd frontend
npm run dev
```

Access the application at: **http://localhost:3000**

## Usage

1. **Open the web interface** at http://localhost:3000
2. **Join a LiveKit room** - The agent will automatically join "voice-fest" room
3. **Start speaking** - The agent detects speech using Silero VAD
4. **View live transcripts** - Transcriptions appear in real-time in the left panel
5. **See AI insights** - Nudges appear in the right panel categorizing content into:
   - Key Proposals
   - Delivery Risks
   - Action Items
6. **Export data** - Click "Export" to download meeting JSON

## Architecture

### Backend Flow

```
LiveKit Room → Agent receives audio
              ↓
         Silero VAD detects speech
              ↓
         MLX Whisper transcribes
              ↓
    FastAPI broadcasts via WebSocket
              ↓
         Frontend receives transcripts
              ↓
    Frontend requests nudges from API
              ↓
         OpenAI classifies content
              ↓
         Nudges displayed in UI
```

### Key Components

- **LiveKit Agent**: Receives real-time audio from meeting participants
- **Silero VAD**: Detects when participants start/stop speaking
- **MLX Whisper**: Transcribes speech locally on Apple Silicon
- **FastAPI Server**: WebSocket server for real-time transcript streaming
- **OpenAI GPT-4**: Classifies transcripts into structured insights
- **Next.js Frontend**: Modern web interface with real-time updates

## API Endpoints

### WebSocket
- `ws://localhost:8000/ws/transcripts` - Real-time transcript stream

### REST API
- `GET /` - Health check
- `POST /api/nudge` - Generate AI insights from transcripts
- `GET /api/meetings` - List all meeting sessions
- `GET /api/meetings/{id}` - Get meeting details
- `POST /api/meetings/{id}/export` - Export meeting data

## Development

### Backend Development

```bash
cd backend
source .venv/bin/activate

# Run with auto-reload
python -m uvicorn api.server:app --reload

# Run agent
python -m agent.main
```

### Frontend Development

```bash
cd frontend
npm run dev        # Development server
npm run build      # Production build
npm run lint       # Lint code
```

## Troubleshooting

### Issue: MLX Whisper model not loading
**Solution**: Ensure you're running on Apple Silicon and MLX is properly installed:
```bash
uv pip install mlx-whisper
```

### Issue: LiveKit connection failed
**Solution**: Verify LiveKit server is running and credentials are correct in `.env`

### Issue: WebSocket not connecting
**Solution**: Ensure FastAPI server is running on port 8000

### Issue: No transcripts appearing
**Solution**:
1. Check LiveKit agent is running
2. Verify participant is in the room
3. Check browser console for errors

## Technologies Used

- **Backend**:
  - LiveKit Agents SDK (Python)
  - Silero VAD
  - MLX Whisper
  - FastAPI
  - OpenAI GPT-4

- **Frontend**:
  - Next.js 14+
  - React
  - TypeScript
  - Tailwind CSS
  - Recharts
  - LiveKit Client SDK

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- LiveKit for the real-time communication platform
- Anthropic for Claude AI assistance in development
- OpenAI for GPT-4 language model
- Apple for MLX framework
