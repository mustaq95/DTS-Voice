import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, cli, AgentServer
from livekit.plugins import silero
import numpy as np
import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config
from agent.transcription import load_whisper_model, transcribe_audio
from agent.vad_handler import VADHandler
from api import storage

# Load environment variables
load_dotenv()

logger = logging.getLogger("meeting-agent")
logger.setLevel(logging.INFO)

# Global server instance
server = AgentServer()

# Global meeting session storage
meeting_sessions = {}


def format_timestamp() -> str:
    """Format current time as HH:MM:SS"""
    return datetime.now().strftime("%H:%M:%S")


async def process_speech_segment(
    participant_identity: str,
    audio_buffer: list,
    meeting_id: str,
    ws_broadcast_fn
):
    """
    Process a complete speech segment after VAD detects speech end

    Args:
        participant_identity: Identity of the speaking participant
        audio_buffer: List of audio frames
        meeting_id: Current meeting session ID
        ws_broadcast_fn: Function to broadcast transcript to WebSocket clients
    """
    if not audio_buffer:
        return

    logger.info(f"Processing speech segment from {participant_identity}, {len(audio_buffer)} frames")

    try:
        # Convert audio buffer to numpy array
        # AudioStream yields AudioFrameEvent objects, which contain a .frame attribute
        audio_data = np.concatenate([
            np.frombuffer(frame.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            for frame in audio_buffer
        ])

        # Transcribe with MLX Whisper
        result = transcribe_audio(audio_data, sample_rate=16000, is_final=True)
        text = result.get("text", "").strip()

        if not text:
            return

        timestamp = format_timestamp()

        # Create transcript entry
        transcript = {
            "timestamp": timestamp,
            "speaker": participant_identity,
            "text": text,
            "is_final": True
        }

        logger.info(f"[{timestamp}] {participant_identity}: {text}")

        # Save to meeting session
        if meeting_id in meeting_sessions:
            session = meeting_sessions[meeting_id]
            storage.add_transcript(
                session,
                timestamp,
                participant_identity,
                text,
                is_final=True
            )
            storage.save_meeting_session(session, config.DATA_DIR)

        # Broadcast to WebSocket clients
        if ws_broadcast_fn:
            await ws_broadcast_fn({
                "type": "transcript",
                "data": transcript
            })

    except Exception as e:
        logger.error(f"Error processing speech segment: {e}")


async def handle_participant_audio(
    participant: rtc.RemoteParticipant,
    meeting_id: str,
    vad: silero.VAD,
    ws_broadcast_fn
):
    """
    Handle audio from a single participant

    Args:
        participant: RemoteParticipant instance
        meeting_id: Meeting session ID
        vad: Silero VAD instance
        ws_broadcast_fn: WebSocket broadcast function
    """
    logger.info(f"Handling audio for participant: {participant.identity}")

    # Create audio buffer for continuous processing
    audio_buffer = []

    async def process_buffer():
        nonlocal audio_buffer
        if len(audio_buffer) > 0:
            await process_speech_segment(
                participant.identity,
                audio_buffer.copy(),
                meeting_id,
                ws_broadcast_fn
            )
            audio_buffer = []

    async def handle_track(track: rtc.Track):
        """Handle incoming audio track"""
        logger.info(f"Subscribed to audio track from {participant.identity}")
        audio_stream = rtc.AudioStream(track)

        async for frame in audio_stream:
            # Buffer all incoming audio frames
            audio_buffer.append(frame)

            # Process buffer when we have ~1.5 seconds of audio
            # Assuming 16kHz sample rate with 20ms frames (50 frames/sec)
            if len(audio_buffer) >= 75:  # ~1.5 seconds of audio
                await process_buffer()

    # Handle existing audio tracks
    for publication in participant.track_publications.values():
        if publication.track and publication.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Found existing audio track for {participant.identity}")
            asyncio.create_task(handle_track(publication.track))


@server.rtc_session(agent_name="meeting-transcriber")
async def entrypoint(ctx: JobContext):
    """
    Main agent entry point - called when agent joins a room

    Args:
        ctx: JobContext containing room, participant, and API access
    """
    logger.info(f"Agent joining room: {ctx.room.name}")

    # Connect to the room
    await ctx.connect()

    room: rtc.Room = ctx.room

    # Create meeting session
    meeting_id = f"{room.name}_{int(datetime.now().timestamp())}"
    session = storage.create_meeting_session(room.name, config.DATA_DIR)
    meeting_sessions[meeting_id] = session
    storage.save_meeting_session(session, config.DATA_DIR)

    logger.info(f"Created meeting session: {meeting_id}")

    # Load Whisper model
    load_whisper_model(config.WHISPER_MODEL)

    # Load VAD
    vad = silero.VAD.load()

    # WebSocket broadcast function (to be connected with FastAPI server)
    # For now, this is a placeholder
    async def ws_broadcast(message):
        # This will be replaced with actual WebSocket broadcast
        logger.info(f"Broadcasting: {message}")

    # Handle existing participants
    for participant in room.remote_participants.values():
        asyncio.create_task(
            handle_participant_audio(participant, meeting_id, vad, ws_broadcast)
        )

    # Handle new participants joining
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")
        asyncio.create_task(
            handle_participant_audio(participant, meeting_id, vad, ws_broadcast)
        )

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant disconnected: {participant.identity}")

    # Handle track subscriptions at room level
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"New audio track subscribed from {participant.identity}")
            # Re-handle audio for this participant to pick up the new track
            asyncio.create_task(
                handle_participant_audio(participant, meeting_id, vad, ws_broadcast)
            )

    # Cleanup callback
    async def cleanup():
        logger.info(f"Cleaning up meeting session: {meeting_id}")
        if meeting_id in meeting_sessions:
            # Save final session state
            storage.save_meeting_session(meeting_sessions[meeting_id], config.DATA_DIR)
            del meeting_sessions[meeting_id]

    ctx.add_shutdown_callback(cleanup)


if __name__ == "__main__":
    # Run the agent server
    cli.run_app(server)
