import asyncio
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, cli, AgentServer
from livekit.plugins import silero
import numpy as np
import os
import sys
from scipy import signal as scipy_signal

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
    room: rtc.Room
):
    """
    Process a complete speech segment after VAD detects speech end

    Args:
        participant_identity: Identity of the speaking participant
        audio_buffer: List of audio frames
        meeting_id: Current meeting session ID
        room: LiveKit Room instance for publishing transcripts via data channel
    """
    if not audio_buffer:
        return

    logger.info(f"Processing speech segment from {participant_identity}, {len(audio_buffer)} frames")

    try:
        # Convert audio buffer to numpy array
        # AudioStream yields AudioFrameEvent objects, which contain a .frame attribute
        # LiveKit audio is typically 48kHz, we need to resample to 16kHz for Whisper
        audio_data_48k = np.concatenate([
            np.frombuffer(frame.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            for frame in audio_buffer
        ])

        # Get the actual sample rate from the first frame
        if audio_buffer:
            source_sample_rate = audio_buffer[0].frame.sample_rate
        else:
            source_sample_rate = 48000  # default assumption

        logger.info(f"Audio data: {len(audio_data_48k)} samples at {source_sample_rate}Hz ({len(audio_data_48k)/source_sample_rate:.2f} seconds)")

        # Resample from source rate to 16kHz for Whisper
        target_sample_rate = 16000
        if source_sample_rate != target_sample_rate:
            num_samples = int(len(audio_data_48k) * target_sample_rate / source_sample_rate)
            audio_data = scipy_signal.resample(audio_data_48k, num_samples)
            logger.debug(f"Resampled audio from {source_sample_rate}Hz to {target_sample_rate}Hz: {len(audio_data)} samples")
        else:
            audio_data = audio_data_48k

        # Transcribe with MLX Whisper
        result = transcribe_audio(audio_data, sample_rate=target_sample_rate, is_final=True)
        text = result.get("text", "").strip()

        if not text:
            logger.debug(f"Empty transcription result from Whisper for {participant_identity}")
            return

        timestamp = format_timestamp()
        logger.info(f"[{timestamp}] {participant_identity}: {text}")

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

        # Publish transcript to LiveKit room via data channel
        try:
            message = {
                "type": "transcript",
                "data": transcript
            }
            payload = json.dumps(message).encode('utf-8')
            await room.local_participant.publish_data(
                payload=payload,
                reliable=True,
                topic="transcription"
            )
            logger.info(f"Published transcript to room via data channel")
        except Exception as pub_error:
            logger.error(f"Error publishing transcript to room: {pub_error}")

    except Exception as e:
        logger.error(f"Error processing speech segment: {e}")


async def handle_participant_audio(
    participant: rtc.RemoteParticipant,
    meeting_id: str,
    vad: silero.VAD,
    room: rtc.Room
):
    """
    Handle audio from a single participant

    Args:
        participant: RemoteParticipant instance
        meeting_id: Meeting session ID
        vad: Silero VAD instance
        room: LiveKit Room instance
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
                room
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

    # Handle existing participants
    for participant in room.remote_participants.values():
        asyncio.create_task(
            handle_participant_audio(participant, meeting_id, vad, room)
        )

    # Handle new participants joining
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")
        asyncio.create_task(
            handle_participant_audio(participant, meeting_id, vad, room)
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
                handle_participant_audio(participant, meeting_id, vad, room)
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
    # For manual testing, you can directly connect to a room:
    # python -m agent.main connect --room voice-fest --url ws://localhost:7880
    cli.run_app(server)
