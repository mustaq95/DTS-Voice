# Claude.md — Realtime Meeting ASR + LiveKit + MLX Whisper Tiny

## Project Overview
I need help in creating this Project from scratch
This project powers a **meeting-room realtime speech-to-text (ASR) agent** for executive discussions.

**Current stage (local validation):**
- LiveKit Agents (Python) receives realtime audio tracks.
- LiveKit Silero VAD detects speech turns.
- MLX Whisper **tiny** transcribes each detected utterance locally on Apple Silicon.
- Agent prints **interim + final transcripts** for verification.

**Next stage (intelligence layer):**
- Final transcripts are pushed into a rolling buffer.
- Separate LLM “nudges” (seperate /nudge API call doing LLM calls) classify content into:
  - Facts 
  - Spotlight
  - Risks 
  - Action Items 
  - Fact-check 

Code Instructions:
- Keep code seperate for frontend, backend, docs , scripts etc.
- Keep code simpler and easy to understand with minimal usage of Classes for Now. This is a POC for now so we need code simpler with explainability
- Use Context7 for latest docs in Livekit
- Use the pasted Image analyze the mockup UI and build the underlying code in this next.js application. Use the recharts library for creating charts to make this a web application. Check how this application looks using the playwright MCP server and verify it looks as close to the mock as possible.


---

## Objectives
1. **Realtime transcription** in a meeting room with low latency.
2. **Robust turn detection** using Silero VAD through LiveKit.
3. **Local inference** on M4 MacBook (no cloud GPU).
4. **Structured realtime insights** via LLM nudges.
5. **Board-ready post-meeting outputs** (summary, decisions, action tracker).

---

## Tech Stack
- **LiveKit Agents SDK (Python)** — realtime audio transport + room/participant handling.
- **LiveKit Silero VAD plugin** — speech/non-speech + turn segmentation.
- **MLX Whisper (tiny model)** — local ASR optimized for Apple Silicon.
- **uv + venv** — dependency management. This is Mandatory
- (Upcoming) LLM services — for nudges and structured meeting intelligence.


