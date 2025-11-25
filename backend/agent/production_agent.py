"""
Production-grade LiveKit agent using best practices
- Proper Silero VAD for speech detection
- MLX Whisper for transcription
- Real-time data channel publishing
- Follows LiveKit Agents framework patterns
"""
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    JobContext,
    JobProcess,
    AgentServer,
    cli,
)
from livekit.agents.vad import VADEventType
from livekit.plugins import silero
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config
from agent.transcription import load_whisper_model, transcribe_audio
import numpy as np
from scipy import signal as scipy_signal

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("production-agent")


def format_timestamp() -> str:
    """Format current time as HH:MM:SS"""
    return datetime.now().strftime("%H:%M:%S")


async def process_audio_with_vad(
    track: rtc.AudioTrack,
    participant_identity: str,
    room: rtc.Room,
    vad: silero.VAD
):
    """
    Process audio track with Silero VAD for proper speech detection.
    Uses VAD to detect speech boundaries and only transcribe complete utterances.

    This follows LiveKit best practices:
    - VAD detects when user starts/stops speaking
    - Audio is buffered during speech
    - Transcription only happens on complete utterances

    Runs two concurrent tasks:
    1. Push audio frames from track to VAD
    2. Process VAD events (START_OF_SPEECH, END_OF_SPEECH)
    """
    logger.info(f"Starting VAD-based audio processing for {participant_identity}")

    # Create VAD stream (one per participant)
    vad_stream = vad.stream()

    # Task 1: Push audio frames from track to VAD
    async def push_audio_frames():
        """Push audio frames to VAD for analysis"""
        audio_stream = rtc.AudioStream(track)
        try:
            async for audio_frame_event in audio_stream:
                # Non-blocking push to VAD's internal channel
                vad_stream.push_frame(audio_frame_event.frame)
        except Exception as e:
            logger.error(f"Error pushing audio frames: {e}")
        finally:
            # Signal end of audio input
            vad_stream.end_input()
            logger.info(f"Audio stream ended for {participant_identity}")

    # Task 2: Process VAD events
    async def process_vad_events():
        """Process VAD events and transcribe complete speech segments"""
        try:
            async for event in vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    logger.info(f"🎙️  Speech started: {participant_identity}")

                elif event.type == VADEventType.END_OF_SPEECH:
                    # VAD detected end of speech - event.frames contains complete segment
                    logger.info(f"✋ Speech ended: {participant_identity}, processing {len(event.frames)} frames")

                    # Transcribe the complete speech segment
                    if event.frames:
                        await transcribe_and_publish(
                            event.frames,
                            participant_identity,
                            room
                        )

        except asyncio.CancelledError:
            logger.info(f"VAD processing cancelled for {participant_identity}")
        except Exception as e:
            logger.error(f"Error processing VAD events: {e}", exc_info=True)
        finally:
            await vad_stream.aclose()

    # Run both tasks concurrently
    await asyncio.gather(
        push_audio_frames(),
        process_vad_events(),
        return_exceptions=True
    )


async def transcribe_and_publish(
    audio_frames: list,
    participant_identity: str,
    room: rtc.Room
):
    """
    Transcribe a complete speech segment and publish to room.

    Args:
        audio_frames: List of AudioFrame objects from VAD
        participant_identity: Speaker identification
        room: LiveKit room for publishing results
    """
    if not audio_frames:
        return

    try:
        # Convert frames to numpy array (LiveKit audio is typically 48kHz)
        audio_data_48k = np.concatenate([
            np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            for frame in audio_frames
        ])

        # Get sample rate from first frame
        source_sample_rate = audio_frames[0].sample_rate if audio_frames else 48000
        target_sample_rate = 16000

        logger.info(f"Audio segment: {len(audio_data_48k)} samples at {source_sample_rate}Hz ({len(audio_data_48k)/source_sample_rate:.2f}s)")

        # Resample to 16kHz for Whisper
        if source_sample_rate != target_sample_rate:
            num_samples = int(len(audio_data_48k) * target_sample_rate / source_sample_rate)
            audio_data = scipy_signal.resample(audio_data_48k, num_samples)
            logger.debug(f"Resampled from {source_sample_rate}Hz to {target_sample_rate}Hz")
        else:
            audio_data = audio_data_48k

        # Transcribe with MLX Whisper
        result = transcribe_audio(audio_data, sample_rate=target_sample_rate, is_final=True)
        text = result.get("text", "").strip()

        if not text:
            logger.debug("Empty transcription, skipping")
            return

        timestamp = format_timestamp()
        logger.info(f"[{timestamp}] {participant_identity}: {text}")

        # Create transcript message
        transcript = {
            "timestamp": timestamp,
            "speaker": participant_identity,
            "text": text,
            "is_final": True
        }

        # Publish to room via data channel
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
        logger.info(f"Published transcript to room")

    except Exception as e:
        logger.error(f"Error in transcribe_and_publish: {e}", exc_info=True)


def prewarm(proc: JobProcess):
    """
    Prewarm function - loads VAD and Whisper models before agent starts.
    This reduces latency on first use.
    """
    logger.info("🔥 Prewarming models...")
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.05,      # 50ms - detect speech start quickly
        min_silence_duration=0.8,      # 800ms - allow natural pauses in sentences
        prefix_padding_duration=0.3,   # 300ms - capture beginning of speech
        max_buffered_speech=60.0,      # 60s - long utterances OK
        activation_threshold=0.5,      # Default threshold
        sample_rate=16000              # Match Whisper's expected rate
    )
    logger.info("✅ VAD model loaded with meeting-optimized settings")


# Create AgentServer (must be at module level for multiprocessing)
server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """
    AgentServer entrypoint - bypasses dispatch and explicitly connects to voice-fest.
    This follows LiveKit patterns but doesn't wait for dispatch.
    """
    logger.info("🚀 AgentServer starting - will connect to voice-fest explicitly")

    # Get pre-warmed VAD
    vad = ctx.proc.userdata["vad"]

    # Load Whisper model
    logger.info(f"📥 Loading Whisper model: {config.WHISPER_MODEL}")
    load_whisper_model(config.WHISPER_MODEL)

    # Create and connect to room explicitly (bypass standard dispatch flow)
    room = rtc.Room()

    # Track audio processing tasks
    participant_tasks = {}

    # Set up event handlers
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"🎵 Audio track subscribed from {participant.identity}")
            task = asyncio.create_task(
                process_audio_with_vad(track, participant.identity, room, vad)
            )
            participant_tasks[participant.identity] = task

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"👤 Participant connected: {participant.identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"👋 Participant disconnected: {participant.identity}")
        if participant.identity in participant_tasks:
            participant_tasks[participant.identity].cancel()
            del participant_tasks[participant.identity]

    # Generate token for agent
    from livekit import api
    token = api.AccessToken(
        api_key=config.LIVEKIT_API_KEY,
        api_secret=config.LIVEKIT_API_SECRET
    )
    token.with_identity("production-agent")
    token.with_name("Production Transcription Agent")
    token.with_grants(api.VideoGrants(
        room_join=True,
        room="voice-fest",
        can_subscribe=True,
        can_publish_data=True,
    ))
    jwt_token = token.to_jwt()

    # Connect to room
    logger.info(f"🔗 Connecting to room: voice-fest")
    await room.connect(config.LIVEKIT_URL, jwt_token)
    logger.info("✅ AgentServer connected with proper VAD!")

    # Keep running (agent server keeps process alive)
    await asyncio.Future()


if __name__ == "__main__":
    # Run with AgentServer CLI - production ready with explicit dispatch
    cli.run_app(server)
