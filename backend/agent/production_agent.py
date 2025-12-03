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
from agent import audio_preprocessing

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import resample_poly

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
        """Push audio frames to VAD with lightweight preprocessing"""
        audio_stream = rtc.AudioStream(track)
        try:
            async for audio_frame_event in audio_stream:
                original_frame = audio_frame_event.frame

                try:
                    # Convert frame data to float32 [-1.0, 1.0]
                    audio_data = np.frombuffer(
                        original_frame.data,
                        dtype=np.int16
                    ).astype(np.float32) / 32768.0

                    # Skip empty frames
                    if len(audio_data) == 0:
                        logger.debug("Skipping empty audio frame")
                        continue

                    # Apply preprocessing pipeline
                    processed_audio = audio_preprocessing.preprocess_audio_frame(
                        audio_data,
                        sample_rate=original_frame.sample_rate,
                        apply_highpass=config.ENABLE_HIGHPASS_FILTER,
                        apply_normalization=config.ENABLE_NORMALIZATION
                    )

                    # Convert back to int16
                    processed_int16 = (processed_audio * 32768.0).astype(np.int16)

                    # Create new AudioFrame with preprocessed audio
                    preprocessed_frame = rtc.AudioFrame(
                        data=processed_int16.tobytes(),
                        sample_rate=original_frame.sample_rate,
                        num_channels=original_frame.num_channels,
                        samples_per_channel=len(processed_int16)
                    )

                    # Push to VAD (gets clean audio)
                    vad_stream.push_frame(preprocessed_frame)

                except ValueError as preprocessing_error:
                    # Log preprocessing failures visibly (don't hide them)
                    logger.error(
                        f"Audio preprocessing failed: {preprocessing_error}. "
                        f"Frame skipped (not using original - we want to see failures)"
                    )
                    # Skip this frame entirely (don't send bad audio to VAD)
                    continue

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
    # Edge case: No audio frames captured
    if not audio_frames:
        logger.debug("No audio frames captured, skipping")
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

        # Edge case: Skip very short segments (< 0.1s)
        duration_seconds = len(audio_data_48k) / source_sample_rate
        if duration_seconds < 0.1:
            logger.debug(f"Audio segment too short ({duration_seconds:.2f}s), skipping transcription")
            return

        logger.info(f"Audio segment: {len(audio_data_48k)} samples at {source_sample_rate}Hz ({duration_seconds:.2f}s)")

        # QUALITY CHECK: Validate audio before transcription (if enabled)
        if config.ENABLE_AUDIO_VALIDATION:
            is_valid, reason = audio_preprocessing.validate_audio_quality(
                audio_data_48k,
                sample_rate=source_sample_rate,
                silence_threshold_db=config.AUDIO_VALIDATION_RMS_THRESHOLD_DB
            )
            if not is_valid:
                logger.warning(f"⚠️  Audio segment rejected: {reason}")
                return

        # Edge case: Check if already at correct sample rate (skip resampling)
        if source_sample_rate != target_sample_rate:
            # Use resample_poly for efficient 48kHz -> 16kHz (3:1 decimation)
            if source_sample_rate == 48000 and target_sample_rate == 16000:
                audio_data = resample_poly(audio_data_48k, 1, 3)  # Downsample by factor of 3
                logger.debug("Resampled from 48kHz to 16kHz using resample_poly (1:3)")
            else:
                # Fallback to general resampling for other rates
                num_samples = int(len(audio_data_48k) * target_sample_rate / source_sample_rate)
                audio_data = scipy_signal.resample(audio_data_48k, num_samples)
                logger.debug(f"Resampled from {source_sample_rate}Hz to {target_sample_rate}Hz")
        else:
            logger.debug(f"Audio already at {target_sample_rate}Hz, skipping resample")
            audio_data = audio_data_48k

        # Transcribe with MLX Whisper
        result = transcribe_audio(audio_data, sample_rate=target_sample_rate, is_final=True)
        text = result.get("text", "").strip()

        if not text:
            logger.debug("Empty transcription, skipping")
            return

        # Detect Whisper hallucinations (repetitive patterns)
        words = text.split()
        if len(words) > 10:
            # Check for excessive word repetition (hallucination indicator)
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # If any single word appears more than 40% of the time, it's likely hallucination
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.4:
                logger.warning(f"⚠️  Rejecting hallucinated transcript: repetitive pattern detected ('{max(word_counts, key=word_counts.get)}' repeated {max_count}/{len(words)} times)")
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

    NOTE: This function is called immediately when the worker starts!
    """
    logger.info("🔥 Prewarming models...")

    # Log preprocessing configuration
    logger.info("🎚️  Audio preprocessing config:")
    if config.ENABLE_HIGHPASS_FILTER:
        logger.info(f"   - High-pass filter: enabled ({config.HIGHPASS_CUTOFF_HZ}Hz)")
    else:
        logger.info("   - High-pass filter: disabled")
    if config.ENABLE_NORMALIZATION:
        logger.info(f"   - Normalization: enabled ({config.NORMALIZATION_TARGET_DB}dB)")
    else:
        logger.info("   - Normalization: disabled")
    if config.ENABLE_AUDIO_VALIDATION:
        logger.info("   - Audio quality validation: enabled (rejects corrupted segments)")
    else:
        logger.info("   - Audio quality validation: disabled")

    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.4,       # Production: balanced sensitivity
        min_speech_duration=0.3,        # Production: reduces false positives
        min_silence_duration=0.6,       # Production: more reliable speech end detection
        prefix_padding_duration=0.3,    # Keep: captures word starts
        max_buffered_speech=60.0,       # Keep as-is
        sample_rate=16000               # Keep as-is
    )

    logger.info("✅ VAD loaded with optimized settings for real-time transcription")


async def entrypoint(ctx: JobContext):
    """
    AgentServer entrypoint - properly connects to assigned room via JobContext.
    This is triggered when a job is dispatched via the API server.
    """
    logger.info("=" * 80)
    logger.info("🚀 AGENT ENTRYPOINT CALLED!")
    logger.info(f"   Room: {ctx.room.name if hasattr(ctx, 'room') else 'pending connection'}")
    logger.info(f"   Job ID: {ctx.job.id if hasattr(ctx, 'job') else 'unknown'}")
    logger.info("=" * 80)

    # Connect to the room provided by JobContext
    await ctx.connect()
    room = ctx.room

    # Wait briefly for room state to stabilize
    await asyncio.sleep(1)

    # Duplicate agent detection: Check if another agent is already in the room
    for participant in room.remote_participants.values():
        if participant.identity.startswith("agent-"):
            logger.warning(f"⚠️  DUPLICATE AGENT DETECTED! Another agent ({participant.identity}) already in room {room.name}")
            logger.warning(f"⚠️  This agent will exit to prevent duplicate processing")
            return  # Exit early - let the existing agent handle this room

    logger.info(f"✅ No duplicate agents detected in room: {room.name}")

    # Get pre-warmed VAD
    vad = ctx.proc.userdata["vad"]

    # Load Whisper model
    logger.info(f"📥 Loading Whisper model: {config.WHISPER_MODEL}")
    load_whisper_model(config.WHISPER_MODEL)

    logger.info(f"✅ Connected to room: {room.name}")

    # Track audio processing tasks
    participant_tasks = {}

    # Create shutdown event to keep agent alive
    shutdown_event = asyncio.Event()

    try:
        # Set up event handlers
        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                # Check if we already have an active task for this participant
                if participant.identity in participant_tasks:
                    existing_task = participant_tasks[participant.identity]
                    if not existing_task.done():
                        logger.info(f"⏭️  Audio track already being processed for {participant.identity}, skipping")
                        return

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

        @room.on("disconnected")
        def on_disconnected():
            logger.info("🔌 Room disconnected, shutting down agent")
            shutdown_event.set()

        logger.info("✅ Agent ready - listening for audio tracks!")

        # Keep agent alive until room disconnects
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Error in agent: {e}", exc_info=True)
    finally:
        # Cleanup: Cancel all participant tasks
        logger.info("🧹 Cleaning up participant tasks...")
        for identity, task in list(participant_tasks.items()):
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if participant_tasks:
            await asyncio.gather(*participant_tasks.values(), return_exceptions=True)

        logger.info("✅ Agent shutdown complete")


if __name__ == "__main__":
    # Run with AgentServer CLI
    from livekit.agents import WorkerOptions, WorkerType

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,  # Auto-join rooms when participants connect
        )
    )
