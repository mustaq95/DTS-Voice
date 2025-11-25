# Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     LiveKit Room                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ User 1   │  │ User 2   │  │  Agent   │                 │
│  │ (audio)  │  │ (audio)  │  │          │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Python Backend (Agent)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Silero VAD   │→ │ MLX Whisper  │→ │   Storage    │     │
│  │ (Speech Det) │  │ (Transcribe) │  │   (JSON)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server                                 │
│  ┌──────────────────────────────────────────────┐          │
│  │          WebSocket Server                     │          │
│  │  /ws/transcripts (real-time stream)          │          │
│  └──────────────────────────────────────────────┘          │
│  ┌──────────────────────────────────────────────┐          │
│  │          REST API                            │          │
│  │  POST /api/nudge (LLM classification)       │          │
│  │  GET  /api/meetings                         │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Next.js Frontend                               │
│  ┌──────────────────────────────────────────────┐          │
│  │  WebSocket Client (receives transcripts)    │          │
│  └──────────────────────────────────────────────┘          │
│  ┌──────────────────────────────────────────────┐          │
│  │  UI Components                               │          │
│  │  - LiveTranscript (left panel)              │          │
│  │  - NudgesPanel (right panel)                │          │
│  │  - MeetingControls (bottom)                 │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Audio Capture & Transcription

```
Participant speaks
    ↓
LiveKit streams audio to Agent
    ↓
Silero VAD detects speech boundaries
    ↓
Audio buffer accumulated during speech
    ↓
MLX Whisper transcribes on speech end
    ↓
Transcript stored in JSON file
    ↓
Broadcast to WebSocket clients
```

### 2. Real-time Updates

```
Backend generates transcript
    ↓
FastAPI WebSocket broadcasts
    ↓
Frontend receives via WebSocket
    ↓
React state updates
    ↓
UI re-renders with new transcript
```

### 3. Nudge Generation

```
Frontend accumulates transcripts
    ↓
Every N transcripts, POST to /api/nudge
    ↓
OpenAI GPT-4 classifies content
    ↓
Returns structured nudges
    ↓
Frontend displays in Nudges panel
```

## Key Technologies

### Backend

**LiveKit Agent (Python)**
- Connects to LiveKit server
- Subscribes to participant audio tracks
- Processes audio in real-time

**Silero VAD**
- Voice Activity Detection
- Speech/silence segmentation
- Turn detection

**MLX Whisper**
- Apple Silicon optimized
- Runs locally (no cloud inference)
- Tiny model for low latency
- ~1-2 second transcription time

**FastAPI**
- WebSocket server for real-time streaming
- REST API for nudge generation
- CORS enabled for Next.js

**OpenAI GPT-4**
- Classifies transcript content
- Extracts key proposals, risks, actions
- Returns confidence scores

### Frontend

**Next.js 14+**
- React Server Components
- App Router
- TypeScript

**WebSocket Client**
- Persistent connection to backend
- Real-time transcript updates

**Recharts**
- Circular progress indicators
- Confidence visualizations

**Tailwind CSS**
- Dark theme (#0f1419, #1a1f2e, #252b3d)
- Responsive design

## File Storage

### Meeting Sessions

Location: `data/meetings/{meeting_id}.json`

```json
{
  "meeting_id": "uuid",
  "room_name": "voice-fest",
  "started_at": "2025-01-15T10:30:00Z",
  "transcripts": [
    {
      "timestamp": "10:30:15",
      "speaker": "Chair",
      "text": "Let's begin the meeting...",
      "is_final": true
    }
  ],
  "nudges": [
    {
      "type": "key_proposal",
      "title": "Budget increase proposal",
      "quote": "I propose a 15% increase",
      "confidence": 0.87
    }
  ]
}
```

## Performance Considerations

### Latency Breakdown

1. **Audio capture**: ~20ms (LiveKit)
2. **VAD processing**: ~10ms
3. **Transcription**: ~1-2s (MLX Whisper Tiny)
4. **WebSocket broadcast**: ~10ms
5. **Frontend render**: ~50ms

**Total**: ~1.3-2.3 seconds from speech end to display

### Optimizations

- **MLX optimization**: Uses Apple Silicon GPU
- **Whisper Tiny model**: Smallest, fastest model
- **Local processing**: No network latency
- **WebSocket**: Lower overhead than HTTP polling
- **React optimizations**: Memoization, efficient re-renders

## Scalability

### Current POC Limitations

- Single agent per room
- JSON file storage (not database)
- No authentication
- Local processing only

### Production Considerations

1. **Database**: Replace JSON with PostgreSQL
2. **Authentication**: Add JWT/session management
3. **Multi-room**: Support multiple concurrent meetings
4. **Distributed**: Deploy agents across multiple servers
5. **CDN**: Serve frontend assets via CDN
6. **Monitoring**: Add metrics, logging, alerting

## Security

### Current Implementation

- CORS restricted to localhost:3000
- No authentication (POC only)
- API keys in .env files
- Local network only

### Production Requirements

- Implement authentication (JWT)
- Add rate limiting
- Use secrets management (not .env)
- Enable HTTPS/WSS
- Add input validation
- Implement access control

## Error Handling

### Backend

- Try-catch blocks around transcription
- WebSocket reconnection logic
- Graceful degradation on API failures

### Frontend

- WebSocket auto-reconnect
- Error boundaries in React
- Fallback UI states
- User-friendly error messages

## Testing

### Backend

```bash
# Unit tests
pytest backend/tests/

# Integration tests
pytest backend/tests/integration/

# Load tests
locust -f backend/tests/load_test.py
```

### Frontend

```bash
# Unit tests
npm test

# E2E tests with Playwright
npm run test:e2e

# Accessibility tests
npm run test:a11y
```

## Deployment

### Development

```bash
./scripts/setup.sh
./scripts/start-backend.sh &
./scripts/start-frontend.sh
```

### Production (Example)

```bash
# Backend (Docker)
docker-compose up -d backend

# Frontend (Vercel)
vercel deploy

# Agent (PM2)
pm2 start backend/agent/main.py --interpreter python3
```

## Monitoring

### Metrics to Track

- Transcription latency
- WebSocket connection count
- API response times
- Nudge generation success rate
- Error rates

### Tools

- Prometheus + Grafana
- Sentry for error tracking
- LiveKit Cloud dashboard
- OpenAI usage dashboard
