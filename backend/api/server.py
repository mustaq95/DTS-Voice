from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Set
import json
import logging
from datetime import datetime
import os
import sys
from livekit import api
import aiohttp

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import storage, llm_service
from agent import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-server")

app = FastAPI(title="LiveKit Meeting Intelligence API")

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections
active_connections: Set[WebSocket] = set()


# Pydantic models for API
class TranscriptRequest(BaseModel):
    meeting_id: str
    timestamp: str
    speaker: str
    text: str
    is_final: bool


class NudgeRequest(BaseModel):
    transcripts: List[str]


class NudgeResponse(BaseModel):
    nudges: List[Dict[str, Any]]


class TokenRequest(BaseModel):
    room_name: str
    participant_name: str


# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "LiveKit Meeting Intelligence API",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/livekit/token")
async def get_livekit_token(request: TokenRequest):
    """
    Generate LiveKit access token for a participant to join a room.
    Also triggers agent dispatch for the room.
    """
    try:
        # Create access token
        token = api.AccessToken(
            api_key=config.LIVEKIT_API_KEY,
            api_secret=config.LIVEKIT_API_SECRET
        )

        # Grant permissions
        token.with_identity(request.participant_name)
        token.with_name(request.participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=request.room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        ))

        # Generate JWT
        jwt_token = token.to_jwt()

        # Explicitly dispatch agent to room
        try:
            from livekit.api.agent_dispatch_service import AgentDispatchService, CreateAgentDispatchRequest

            async with aiohttp.ClientSession() as session:
                dispatch_service = AgentDispatchService(
                    session,
                    config.LIVEKIT_URL,
                    config.LIVEKIT_API_KEY,
                    config.LIVEKIT_API_SECRET
                )
                await dispatch_service.create_dispatch(
                    CreateAgentDispatchRequest(
                        room=request.room_name,
                        agent_name="",  # Empty = any available agent
                        metadata="transcription-agent"
                    )
                )
            logger.info(f"✅ Agent dispatched to room: {request.room_name}")
        except Exception as dispatch_error:
            logger.warning(f"⚠️  Agent dispatch failed (may already be connected): {dispatch_error}")

        return {
            "token": jwt_token,
            "url": config.LIVEKIT_URL
        }

    except Exception as e:
        logger.error(f"Error generating LiveKit token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/transcripts")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming transcripts to frontend
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive client messages
            data = await websocket.receive_text()
            logger.info(f"Received from client: {data}")

            # Echo back or handle client commands
            # This could be used for client → backend commands

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.post("/api/transcripts/broadcast")
async def broadcast_transcript(transcript: TranscriptRequest):
    """
    Endpoint for agent to broadcast new transcripts to all connected clients

    This is called by the LiveKit agent when new transcripts are available
    """
    message = {
        "type": "transcript",
        "data": {
            "timestamp": transcript.timestamp,
            "speaker": transcript.speaker,
            "text": transcript.text,
            "is_final": transcript.is_final
        }
    }

    await manager.broadcast(message)

    return {"status": "broadcasted", "connections": len(manager.active_connections)}


@app.post("/api/nudge", response_model=NudgeResponse)
async def generate_nudges(request: NudgeRequest):
    """
    Generate LLM-based nudges/insights from transcripts

    This endpoint classifies transcript content into:
    - Key Proposals
    - Delivery Risks
    - Action Items
    """
    try:
        nudges = llm_service.classify_transcripts(
            request.transcripts,
            config.OPENAI_API_KEY
        )

        return {"nudges": nudges}

    except Exception as e:
        logger.error(f"Error generating nudges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meetings")
async def list_meetings():
    """List all meeting sessions"""
    import os
    import glob

    meetings_path = os.path.join(config.DATA_DIR, "meetings", "*.json")
    meeting_files = glob.glob(meetings_path)

    meetings = []
    for file_path in meeting_files:
        try:
            with open(file_path, 'r') as f:
                meeting = json.load(f)
                meetings.append({
                    "meeting_id": meeting["meeting_id"],
                    "room_name": meeting["room_name"],
                    "started_at": meeting["started_at"],
                    "transcript_count": len(meeting["transcripts"]),
                    "nudge_count": len(meeting["nudges"])
                })
        except Exception as e:
            logger.error(f"Error loading meeting {file_path}: {e}")

    return {"meetings": meetings}


@app.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """Get detailed meeting data"""
    try:
        meeting = storage.load_meeting_session(meeting_id, config.DATA_DIR)
        return meeting
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Meeting not found")
    except Exception as e:
        logger.error(f"Error loading meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/meetings/{meeting_id}/export")
async def export_meeting(meeting_id: str):
    """Export meeting data as JSON"""
    try:
        meeting = storage.load_meeting_session(meeting_id, config.DATA_DIR)

        # Return the complete meeting data
        return {
            "meeting_id": meeting["meeting_id"],
            "room_name": meeting["room_name"],
            "started_at": meeting["started_at"],
            "transcripts": meeting["transcripts"],
            "nudges": meeting["nudges"]
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Meeting not found")
    except Exception as e:
        logger.error(f"Error exporting meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility function for agents to broadcast via WebSocket
async def broadcast_to_websocket(message: dict):
    """
    Utility function that can be imported by the agent
    to broadcast messages to WebSocket clients
    """
    await manager.broadcast(message)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
