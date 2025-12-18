"""
Production-grade LiveKit agent using best practices
- Proper Silero VAD for speech detection
- MLX Whisper for transcription
- Hamza Streaming STT for Arabic (production-optimized)
- Real-time data channel publishing
- Follows LiveKit Agents framework patterns
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
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
from agent.hamza_transcription import (
    transcribe_audio_hamza,
    close_hamza_client,
    get_hamza_client,
    HamzaStreamingClient,
    HamzaLiveKitStreamer
)
from agent import audio_preprocessing
from agent.segment_manager import SegmentManager
from agent.llm_classifier_worker import LLMClassifierClient

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import resample_poly

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("production-agent")

# Global state: Track segment managers per meeting
segment_managers = {}

# Global state: Current transcription engine per room
room_transcription_engines = {}  # {"room_name": "mlx_whisper" | "hamza"}

# Global state: Hamza streamers for real-time mode
hamza_streamers = {}  # {"room_name": HamzaLiveKitStreamer}


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

    # Track background transcription tasks to prevent speech loss
    transcription_tasks = []

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

                    # PERFORMANCE FIX: Resample to 16kHz BEFORE preprocessing
                    # This makes VAD 3x faster (processes 16kHz instead of 48kHz)
                    source_sample_rate = original_frame.sample_rate
                    target_sample_rate = 16000  # VAD is configured for 16kHz

                    if source_sample_rate != target_sample_rate:
                        # Efficient downsampling: 48kHz → 16kHz (3:1 decimation)
                        if source_sample_rate == 48000 and target_sample_rate == 16000:
                            audio_data = resample_poly(audio_data, 1, 3)
                        else:
                            # Fallback for other sample rates
                            num_samples = int(len(audio_data) * target_sample_rate / source_sample_rate)
                            audio_data = scipy_signal.resample(audio_data, num_samples)

                    # Apply preprocessing pipeline at 16kHz (3x faster than at 48kHz)
                    processed_audio = audio_preprocessing.preprocess_audio_frame(
                        audio_data,
                        sample_rate=target_sample_rate,
                        apply_highpass=config.ENABLE_HIGHPASS_FILTER,
                        apply_normalization=config.ENABLE_NORMALIZATION
                    )

                    # Convert back to int16
                    processed_int16 = (processed_audio * 32768.0).astype(np.int16)

                    # Create new AudioFrame at 16kHz for VAD (no internal resampling needed)
                    preprocessed_frame = rtc.AudioFrame(
                        data=processed_int16.tobytes(),
                        sample_rate=target_sample_rate,
                        num_channels=original_frame.num_channels,
                        samples_per_channel=len(processed_int16)
                    )

                    # Push to VAD (now processes at realtime speed)
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

                    # Transcribe the complete speech segment (non-blocking to prevent speech loss)
                    if event.frames:
                        task = asyncio.create_task(
                            transcribe_and_publish(
                                event.frames,
                                participant_identity,
                                room
                            )
                        )
                        transcription_tasks.append(task)
                        # Cleanup completed tasks to prevent memory leak
                        transcription_tasks[:] = [t for t in transcription_tasks if not t.done()]

        except asyncio.CancelledError:
            logger.info(f"VAD processing cancelled for {participant_identity}")
        except Exception as e:
            logger.error(f"Error processing VAD events: {e}", exc_info=True)
        finally:
            await vad_stream.aclose()
            # Wait for any pending transcriptions to complete before shutting down
            if transcription_tasks:
                logger.info(f"Waiting for {len(transcription_tasks)} pending transcriptions to complete...")
                await asyncio.gather(*transcription_tasks, return_exceptions=True)

    # Run both tasks concurrently
    await asyncio.gather(
        push_audio_frames(),
        process_vad_events(),
        return_exceptions=True
    )


def on_data_received(data_packet: rtc.DataPacket):
    """
    Handle incoming data messages from LiveKit room.

    Used for receiving model switch commands from API server.
    """
    try:
        # DEBUG: Log ALL data packets received
        sender_id = data_packet.participant.identity if data_packet.participant else 'unknown'
        logger.info(f"📨 Data packet received - Sender: {sender_id}, Kind: {data_packet.kind}, Size: {len(data_packet.data)} bytes")

        # Decode message
        message = json.loads(data_packet.data.decode('utf-8'))
        logger.info(f"📨 Decoded message: {message}")
        msg_type = message.get("type")

        if msg_type == "model_switch":
            # Extract model switch data
            data = message.get("data", {})
            new_model = data.get("model")
            room_name = data.get("room_name")

            if new_model and room_name:
                # Get previous model
                previous_model = room_transcription_engines.get(room_name, config.TRANSCRIPTION_ENGINE)

                # Update agent's local dict
                room_transcription_engines[room_name] = new_model

                logger.info(f"🔄 Model switched via data channel: {previous_model} → {new_model} (room: {room_name})")

                # Close Hamza connection if switching AWAY from Hamza
                if previous_model == "hamza" and new_model != "hamza":
                    logger.info("🧹 Closing Hamza WebSocket (switched to different model)")
                    asyncio.create_task(_close_hamza_on_switch(room_name))

                # Pre-initialize Hamza if switching TO Hamza
                if new_model == "hamza":
                    logger.info("🔧 Pre-initializing Hamza WebSocket after model switch...")
                    asyncio.create_task(_init_hamza_client())
            else:
                logger.warning(f"⚠️  Invalid model_switch message: {message}")
        else:
            logger.debug(f"📨 Received non-model_switch message type: {msg_type}")

    except json.JSONDecodeError as e:
        logger.warning(f"⚠️  Non-JSON data packet received: {e} - Raw data: {data_packet.data[:100]}")
    except Exception as e:
        logger.error(f"❌ Error handling data packet: {e}", exc_info=True)


async def _init_hamza_client():
    """Helper to initialize Hamza client in background"""
    try:
        hamza_client = await get_hamza_client()
        if hamza_client:
            logger.info("✅ Hamza WebSocket ready after model switch")
        else:
            logger.error("❌ Failed to initialize Hamza after model switch")
    except Exception as e:
        logger.error(f"❌ Hamza initialization failed: {e}", exc_info=True)


async def _close_hamza_on_switch(room_name: str):
    """Helper to close Hamza client when switching away from Hamza"""
    try:
        # Close room-specific streamer if exists
        if room_name in hamza_streamers:
            await hamza_streamers[room_name].close()
            del hamza_streamers[room_name]
            logger.info(f"✅ Hamza streamer closed for room: {room_name}")
        
        # Close global client if no more Hamza rooms
        hamza_rooms = [r for r, e in room_transcription_engines.items() if e == "hamza"]
        if not hamza_rooms:
            await close_hamza_client()
            logger.info("✅ Hamza WebSocket closed (no Hamza rooms)")
    except Exception as e:
        logger.error(f"❌ Error closing Hamza after switch: {e}", exc_info=True)


async def transcribe_hamza_streaming(
    audio_data: np.ndarray,
    sample_rate: int,
    room: rtc.Room,
    participant_identity: str
) -> Dict[str, Any]:
    """
    Transcribe with Hamza using real-time streaming - publishes partial results as they arrive.

    Args:
        audio_data: Audio samples (float32)
        sample_rate: Sample rate
        room: LiveKit room for publishing
        participant_identity: Speaker ID

    Returns:
        dict: Final transcription result with metrics (timestamp included for deduplication)
    """
    from agent.hamza_transcription import get_hamza_client
    import json

    try:
        client = await get_hamza_client()
        if client is None:
            return {
                "text": "",
                "error": "Hamza client not available",
                "engine": "hamza"
            }

        # Start streaming
        await client.start_streaming()

        # Publish each fragment immediately as it arrives
        last_partial_text = ""
        fragment_count = 0

        async def on_partial(text: str):
            nonlocal last_partial_text, fragment_count
            if text and text != last_partial_text:
                last_partial_text = text
                fragment_count += 1

                logger.info(f"📝 Fragment #{fragment_count}: {text[:50]}...")

                # Publish fragment immediately to UI
                timestamp = format_timestamp()
                partial_transcript = {
                    "timestamp": timestamp,
                    "speaker": participant_identity,
                    "text": text,
                    "is_final": True,  # Each fragment is complete by itself
                    "engine": "hamza"
                }

                message = {
                    "type": "transcript",
                    "data": partial_transcript
                }
                payload = json.dumps(message).encode('utf-8')
                await room.local_participant.publish_data(payload, reliable=True)

        # Set callback
        client.on_partial_result = on_partial

        # Send audio in chunks
        chunk_size = 512  # 32ms at 16kHz
        chunk_delay = 0.015  # 15ms between chunks

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            await client.send_audio_chunk(chunk)
            await asyncio.sleep(chunk_delay)

        # Stop streaming
        await client.stop_streaming()

        logger.info(f"✅ Published {fragment_count} fragments in real-time")

        # Return empty result - fragments already published via callback
        return {
            "text": "",  # Already published fragments, skip final publish
            "is_final": True,
            "language": "ar",
            "engine": "hamza",
            "metrics": client.metrics.to_dict(),
            "skip_publish": True  # Flag to skip duplicate publishing
        }

    except Exception as e:
        logger.error(f"❌ Hamza streaming error: {e}", exc_info=True)
        return {
            "text": "",
            "error": str(e),
            "engine": "hamza"
        }


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
        # Convert frames to numpy array
        audio_data_source = np.concatenate([
            np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            for frame in audio_frames
        ])

        # Get sample rate from first frame
        source_sample_rate = audio_frames[0].sample_rate if audio_frames else 16000
        target_sample_rate = 16000

        # Calculate duration
        duration_seconds = len(audio_data_source) / source_sample_rate

        # PERMISSIVE: Only skip extremely short segments (< 0.2s) - mostly just clicks
        if duration_seconds < 0.2:
            logger.debug(f"Audio segment too short ({duration_seconds:.2f}s), skipping transcription")
            return

        logger.info(f"Audio segment: {len(audio_data_source)} samples at {source_sample_rate}Hz ({duration_seconds:.2f}s)")

        # QUALITY CHECK: Validate audio before transcription (if enabled)
        if config.ENABLE_AUDIO_VALIDATION:
            is_valid, reason = audio_preprocessing.validate_audio_quality(
                audio_data_source,
                sample_rate=source_sample_rate,
                silence_threshold_db=config.AUDIO_VALIDATION_RMS_THRESHOLD_DB
            )
            if not is_valid:
                logger.warning(f"⚠️  Audio segment rejected: {reason}")
                return

        # Resample to 16kHz if needed
        if source_sample_rate != target_sample_rate:
            if source_sample_rate == 48000 and target_sample_rate == 16000:
                audio_data = resample_poly(audio_data_source, 1, 3)
                logger.debug("Resampled from 48kHz to 16kHz using resample_poly (1:3)")
            else:
                num_samples = int(len(audio_data_source) * target_sample_rate / source_sample_rate)
                audio_data = scipy_signal.resample(audio_data_source, num_samples)
                logger.debug(f"Resampled from {source_sample_rate}Hz to {target_sample_rate}Hz")
        else:
            audio_data = audio_data_source

        # Check current transcription engine for this room (read from file for latest config)
        from agent.model_config import load_model_config
        current_engine = load_model_config(room.name, config.TRANSCRIPTION_ENGINE)

        # Update local cache
        room_transcription_engines[room.name] = current_engine

        logger.info(f"Using transcription engine: {current_engine}")

        # Transcribe with selected engine
        if current_engine == "hamza":
            # Use streaming mode with real-time partial results
            result = await transcribe_hamza_streaming(
                audio_data=audio_data,
                sample_rate=target_sample_rate,
                room=room,
                participant_identity=participant_identity
            )

            # If Hamza fails, DON'T fallback to MLX - just skip this transcript
            if result.get("error"):
                logger.error(f"❌ Hamza transcription failed: {result['error']}")
                return

            # Log metrics if available
            if "metrics" in result:
                metrics = result["metrics"]
                logger.info(
                    f"📊 Hamza metrics: latency={metrics.get('avg_latency_ms', 0):.0f}ms, "
                    f"TTFR={metrics.get('time_to_first_result_ms', 0):.0f}ms"
                )

            # Check if fragments were already published in real-time
            if result.get("skip_publish"):
                logger.info("✅ Hamza fragments already published in real-time, skipping final publish")
                # Still add to segment manager for classification
                # Note: segment manager will need to handle multiple fragment transcripts
                return

        else:
            # Use MLX Whisper
            result = transcribe_audio(audio_data, sample_rate=target_sample_rate, is_final=True)
            result["engine"] = "mlx_whisper"

        text = result.get("text", "").strip()
        engine_used = result.get('engine', current_engine)
        logger.info(f"Transcription completed using: {engine_used}")

        if not text:
            logger.debug("Empty transcription, skipping")
            return

        # BALANCED: Detect obvious hallucinations (35% threshold)
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_lower = word.lower().strip('.,!?')
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

            max_count = max(word_counts.values())
            if max_count > len(words) * 0.35:
                most_repeated = max(word_counts, key=word_counts.get)
                logger.warning(
                    f"⚠️  Rejecting hallucinated transcript: "
                    f"'{most_repeated}' repeated {max_count}/{len(words)} times"
                )
                return

        # Additional hallucination patterns
        HALLUCINATION_PHRASES = [
            "thank you for watching",
            "thanks for watching",
            "subscribe",
            "like and subscribe",
            "see you next time",
            "bye bye bye",
        ]
        text_lower = text.lower()
        for phrase in HALLUCINATION_PHRASES:
            if phrase in text_lower:
                logger.warning(f"⚠️  Rejecting hallucination: contains '{phrase}'")
                return

        timestamp = format_timestamp()
        logger.info(f"[{timestamp}] {participant_identity}: {text}")

        # Create transcript message
        transcript = {
            "timestamp": timestamp,
            "speaker": participant_identity,
            "text": text,
            "is_final": True,
            "engine": engine_used
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

        # Add to segment manager for topic segmentation
        if room.name in segment_managers:
            logger.info(f"📤 Adding transcript to SegmentManager for classification")
            manager = segment_managers[room.name]
            manager.add_transcript(
                timestamp=timestamp,
                text=text,
                speaker=participant_identity
            )
            logger.info(f"✅ Transcript added to SegmentManager (will be processed asynchronously)")
        else:
            logger.warning(f"⚠️  No SegmentManager found for room {room.name}, skipping classification")

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

    # Log transcription engine
    logger.info(f"🎙️  Default transcription engine: {config.TRANSCRIPTION_ENGINE}")

    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.35,      # Lower threshold for softer speech
        min_speech_duration=0.25,       # Faster detection start
        min_silence_duration=1.2,       # Longer tolerance for natural pauses
        prefix_padding_duration=1.0,    # INCREASED: 1 second padding to capture full beginning
        max_buffered_speech=60.0,       # Max 60 seconds per utterance
        sample_rate=16000               # Match Whisper/Hamza sample rate
    )

    logger.info("✅ VAD loaded with optimized settings")

    # Create LLM classifier client (runs in separate process for CPU isolation)
    logger.info("🔧 Creating LLM classifier client (separate process)...")
    classifier_client = LLMClassifierClient()
    classifier_client.start()
    logger.info("✅ LLM classifier worker process started")
    proc.userdata["classifier_client"] = classifier_client


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
            return

    logger.info(f"✅ No duplicate agents detected in room: {room.name}")

    # Get pre-warmed models
    vad = ctx.proc.userdata["vad"]
    classifier_client = ctx.proc.userdata["classifier_client"]

    # Load Whisper model (for MLX engine)
    if config.TRANSCRIPTION_ENGINE == "mlx_whisper":
        logger.info(f"📥 Loading Whisper model: {config.WHISPER_MODEL}")
        load_whisper_model(config.WHISPER_MODEL)

    logger.info(f"✅ Connected to room: {room.name}")

    # Create segment manager for this meeting
    segment_managers[room.name] = SegmentManager(
        meeting_id=room.name,
        room=room,
        classifier_client=classifier_client
    )
    logger.info(f"✅ SegmentManager created for room: {room.name}")

    # Initialize room transcription engine (default from config)
    room_transcription_engines[room.name] = config.TRANSCRIPTION_ENGINE
    logger.info(f"🎙️  Transcription engine set to: {room_transcription_engines[room.name]}")

    # Pre-initialize Hamza if configured
    if config.TRANSCRIPTION_ENGINE == "hamza":
        logger.info("🔧 Pre-initializing Hamza WebSocket...")
        try:
            hamza_client = await get_hamza_client()
            if hamza_client:
                logger.info("✅ Hamza WebSocket ready")
            else:
                logger.error("❌ Failed to initialize Hamza - transcription may fail")
        except Exception as e:
            logger.error(f"❌ Hamza pre-initialization failed: {e}")

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

        # Register data message handler for model switching
        room.on("data_received", on_data_received)
        logger.info("✅ Data message handler registered (for model switching)")

        # Subscribe to existing participants' audio tracks
        for participant in room.remote_participants.values():
            logger.info(f"👤 Found existing participant: {participant.identity}")
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"🎵 Found existing audio track from {participant.identity}")

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

        # Cleanup segment manager
        if room.name in segment_managers:
            logger.info("🧹 Cleaning up segment manager...")
            await segment_managers[room.name].cleanup()
            del segment_managers[room.name]

        # Cleanup Hamza streamer if exists
        if room.name in hamza_streamers:
            logger.info("🧹 Cleaning up Hamza streamer...")
            await hamza_streamers[room.name].close()
            del hamza_streamers[room.name]

        # Cleanup room transcription engine state
        if room.name in room_transcription_engines:
            del room_transcription_engines[room.name]

        # Cleanup model config file
        from agent.model_config import delete_model_config
        delete_model_config(room.name)
        logger.info(f"🗑️  Deleted model config for room: {room.name}")

        # Close Hamza WebSocket if this was the last room
        if len(room_transcription_engines) == 0:
            logger.info("🧹 Closing Hamza WebSocket (no active rooms)")
            await close_hamza_client()

        logger.info("✅ Agent shutdown complete")


if __name__ == "__main__":
    # Run with AgentServer CLI
    from livekit.agents import WorkerOptions, WorkerType

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            agent_name="production-agent",
        )
    )