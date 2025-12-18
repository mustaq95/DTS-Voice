from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Set
import json
import logging
from datetime import datetime
import os
import sys
import aiohttp
import asyncio
from livekit import api
from livekit.api.agent_dispatch_service import AgentDispatchService

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

# Track rooms where agents have been dispatched
dispatched_rooms: set[str] = set()


# Pydantic models for API
class TranscriptRequest(BaseModel):
    meeting_id: str
    timestamp: str
    speaker: str
    text: str
    is_final: bool


class NudgeRequest(BaseModel):
    transcripts: List[str]
    segment_id: str | None = None  # Optional segment ID for context
    topic: str | None = None  # Optional topic for better classification


class NudgeResponse(BaseModel):
    nudges: List[Dict[str, Any]]


class TokenRequest(BaseModel):
    room_name: str
    participant_name: str


class ModelConfigRequest(BaseModel):
    room_name: str
    model: str  # "mlx_whisper" or "hamza"


class ModelConfigResponse(BaseModel):
    status: str
    room_name: str
    previous_model: str
    current_model: str
    message: str


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
    Agent auto-joins via WorkerType.ROOM - no manual dispatch needed.
    """
    try:
        token = api.AccessToken(
            api_key=config.LIVEKIT_API_KEY,
            api_secret=config.LIVEKIT_API_SECRET
        )

        token.with_identity(request.participant_name)
        token.with_name(request.participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=request.room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        ))

        jwt_token = token.to_jwt()

        logger.info(f"✅ Token generated for {request.participant_name} in room {request.room_name}")

        # Dispatch agent to room if not already dispatched
        if request.room_name not in dispatched_rooms:
            try:
                # Convert ws:// to http:// for dispatch service URL
                dispatch_url = config.LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")

                # # Create agent dispatch service
                # dispatch_service = AgentDispatchService(
                #     api_key=config.LIVEKIT_API_KEY,
                #     api_secret=config.LIVEKIT_API_SECRET,
                #     url=dispatch_url
                # )

                # # Dispatch agent to room
                # await dispatch_service.create_dispatch(
                #     room=request.room_name,
                #     agent_name="production-agent"
                # )

                # Use LiveKitAPI (this sets up AgentDispatchService with a session)
                lkapi = api.LiveKitAPI(
                    url=dispatch_url,
                    api_key=config.LIVEKIT_API_KEY,
                    api_secret=config.LIVEKIT_API_SECRET,
                )

                # Create explicit dispatch
                await lkapi.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(
                        agent_name="production-agent",   # must match your WorkerOptions.agentName
                        room=request.room_name,
                        # optional metadata
                        metadata=json.dumps({"requested_by": request.participant_name}),
                    )
                )

                await lkapi.aclose()

                dispatched_rooms.add(request.room_name)
                logger.info(f"🤖 Agent dispatched to room: {request.room_name}")

            except Exception as dispatch_error:
                logger.warning(f"Agent dispatch failed (non-critical): {dispatch_error}")
                # Don't fail token generation if dispatch fails

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

    Now optimized to work with segment-based context for better semantic analysis.
    """
    try:
        # Log segment context if provided
        if request.segment_id and request.topic:
            logger.info(f"📊 Generating nudges for segment '{request.topic}' (ID: {request.segment_id})")

        nudges = llm_service.classify_transcripts(
            transcripts=request.transcripts,
            api_key=config.OPENAI_API_KEY,
            topic=request.topic  # Pass topic for enhanced context
        )

        # Add segment_id to each nudge for tracking
        if request.segment_id:
            for nudge in nudges:
                nudge['segment_id'] = request.segment_id

        logger.info(f"✅ Generated {len(nudges)} nudges")
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


@app.get("/api/meetings/{meeting_id}/state")
async def get_meeting_state(meeting_id: str):
    """
    Get current segmentation state for a meeting.
    Returns completed segments + active buffer.
    """
    try:
        # Read from data/meetings/{meeting_id}/segments.json
        segments_file = os.path.join(
            config.DATA_DIR,
            "meetings",
            meeting_id,
            "segments.json"
        )

        if os.path.exists(segments_file):
            with open(segments_file, 'r') as f:
                return json.load(f)
        else:
            # No segments file yet - return empty state
            return {
                "meeting_id": meeting_id,
                "segments": [],
                "current_segment": None
            }

    except Exception as e:
        logger.error(f"Error getting meeting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meetings/{meeting_id}/segments")
async def get_segments(meeting_id: str):
    """
    Get only completed segments (no active buffer).
    Useful for displaying finalized topics.
    """
    try:
        state = await get_meeting_state(meeting_id)
        return {"segments": state.get("segments", [])}

    except Exception as e:
        logger.error(f"Error getting segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meetings/{meeting_id}/buffer")
async def get_buffer(meeting_id: str):
    """
    Get current active segment buffer.
    Shows what's being actively transcribed/segmented.
    """
    try:
        state = await get_meeting_state(meeting_id)
        return state.get("current_segment", None)

    except Exception as e:
        logger.error(f"Error getting buffer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/api/redeploy")
async def redeploy_system():
    """
    Redeploy the entire system by running stop.sh and start.sh scripts.
    This restarts all services (LiveKit, backend, frontend) from scratch.

    The restart happens in a detached background process to survive the API shutdown.
    """
    try:
        # Get the project root directory (parent of backend)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        stop_script = os.path.join(project_root, "stop.sh")
        start_script = os.path.join(project_root, "start.sh")

        logger.info("🔄 Starting system redeployment...")

        # Check if scripts exist
        if not os.path.exists(stop_script):
            raise HTTPException(status_code=404, detail=f"stop.sh not found at {stop_script}")
        if not os.path.exists(start_script):
            raise HTTPException(status_code=404, detail=f"start.sh not found at {start_script}")

        # Create a shell script that will run stop.sh and start.sh in sequence
        # This needs to run in a completely detached process that survives after this API dies
        restart_command = f'''
        nohup bash -c '
            cd {project_root}
            echo "🔄 Redeployment initiated at $(date)" >> /tmp/redeploy.log

            # Stop all services
            echo "Stopping services..." >> /tmp/redeploy.log
            bash {stop_script} >> /tmp/redeploy.log 2>&1

            # Wait for complete shutdown
            sleep 3

            # Start all services
            echo "Starting services..." >> /tmp/redeploy.log
            bash {start_script} >> /tmp/redeploy.log 2>&1

            echo "✅ Redeployment completed at $(date)" >> /tmp/redeploy.log
        ' > /dev/null 2>&1 &
        '''

        # Execute the detached restart process
        logger.info("Launching detached restart process...")
        await asyncio.create_subprocess_shell(
            restart_command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True  # Detach from parent process
        )

        logger.info("✅ Redeployment initiated in background")

        return {
            "status": "success",
            "message": "System redeployment initiated. Services are restarting...",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error during redeployment: {e}")
        raise HTTPException(status_code=500, detail=f"Redeployment failed: {str(e)}")


@app.post("/api/config/model", response_model=ModelConfigResponse)
async def switch_transcription_model(request: ModelConfigRequest):
    """Switch transcription model for a room at runtime (mid-session)."""
    try:
        # Validate model
        valid_models = ["mlx_whisper", "hamza"]
        if request.model not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Must be one of: {valid_models}"
            )

        # Get previous model (from config default)
        previous_model = config.TRANSCRIPTION_ENGINE

        # Send model switch command via LiveKit data channel
        # The agent process will receive this and update its local dict
        dispatch_url = config.LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")

        lkapi = api.LiveKitAPI(
            url=dispatch_url,
            api_key=config.LIVEKIT_API_KEY,
            api_secret=config.LIVEKIT_API_SECRET,
        )

        # Send data message to room with model switch command
        message = {
            "type": "model_switch",
            "data": {
                "model": request.model,
                "room_name": request.room_name
            }
        }

        # Use SendDataRequest to send data to room
        send_request = api.SendDataRequest(
            room=request.room_name,
            data=json.dumps(message).encode('utf-8'),
            destination_identities=[],  # Broadcast to all participants (agent will receive)
        )

        await lkapi.room.send_data(send_request)

        await lkapi.aclose()

        logger.info(f"🔄 Model switch command sent to room: {request.room_name} → {request.model}")

        # IMPORTANT: Save to file for agent to read (data messages don't reach agents)
        from agent.model_config import save_model_config
        save_model_config(request.room_name, request.model)
        logger.info(f"💾 Saved model config to file: {request.room_name} → {request.model}")

        # Pre-initialize Hamza WebSocket in API server (optional, for health checks)
        if request.model == "hamza":
            from agent.hamza_transcription import get_hamza_client
            try:
                hamza_client = await get_hamza_client()
                if hamza_client is None:
                    logger.warning("⚠️  Hamza client initialization failed in API server")
            except Exception as e:
                logger.warning(f"⚠️  Hamza initialization error: {e}")

        return ModelConfigResponse(
            status="success",
            room_name=request.room_name,
            previous_model=previous_model,
            current_model=request.model,
            message=f"Model switch command sent: '{request.model}'"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending model switch command: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config/model/{room_name}")
async def get_current_model(room_name: str):
    """Get current transcription model for a room."""
    try:
        # Return default from config (agent initializes with this)
        # Note: After model switches, this won't reflect the actual state
        # For accurate state, would need to query room metadata
        return {
            "room_name": room_name,
            "current_model": config.TRANSCRIPTION_ENGINE,
            "note": "Default model - may not reflect mid-session switches"
        }
    except Exception as e:
        logger.error(f"Error getting current model: {e}")
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
