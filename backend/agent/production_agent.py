"""
Production-grade LiveKit agent using best practices
- Proper Silero VAD for speech detection
- MLX Whisper for transcription
- Hamza Streaming STT for Arabic (production-optimized)
- Real-time data channel publishing
- Follows LiveKit Agents framework patterns

FIXED: First speech empty transcription issue
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from livekit import rtc
from livekit.rtc import Room
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

# Global state: Nudge context buffer and generated nudges per room
nudge_context_buffer = {}  # {"room_name": [last 10 transcripts]}
generated_nudges = {}  # {"room_name": [all generated nudges]}

# Global state: Track VAD streams per participant for manual flushing
participant_vad_streams = {}  # {"participant_identity": vad_stream}


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
    """
    logger.info(f"Starting VAD-based audio processing for {participant_identity}")

    vad_stream = vad.stream()
    transcription_tasks = []

    # Store VAD stream for manual flushing (when mic turned off)
    participant_vad_streams[participant_identity] = vad_stream

    async def push_audio_frames():
        """Push audio frames to VAD with preprocessing"""
        audio_stream = rtc.AudioStream(track)
        try:
            async for audio_frame_event in audio_stream:
                original_frame = audio_frame_event.frame

                try:
                    audio_data = np.frombuffer(
                        original_frame.data,
                        dtype=np.int16
                    ).astype(np.float32) / 32768.0

                    if len(audio_data) == 0:
                        continue

                    source_sample_rate = original_frame.sample_rate
                    target_sample_rate = 16000

                    if source_sample_rate != target_sample_rate:
                        if source_sample_rate == 48000:
                            audio_data = resample_poly(audio_data, 1, 3)
                        else:
                            num_samples = int(len(audio_data) * target_sample_rate / source_sample_rate)
                            audio_data = scipy_signal.resample(audio_data, num_samples)

                    processed_audio = audio_preprocessing.preprocess_audio_frame(
                        audio_data,
                        sample_rate=target_sample_rate,
                        apply_highpass=config.ENABLE_HIGHPASS_FILTER,
                        apply_normalization=config.ENABLE_NORMALIZATION
                    )

                    processed_int16 = (processed_audio * 32768.0).astype(np.int16)

                    preprocessed_frame = rtc.AudioFrame(
                        data=processed_int16.tobytes(),
                        sample_rate=target_sample_rate,
                        num_channels=original_frame.num_channels,
                        samples_per_channel=len(processed_int16)
                    )

                    vad_stream.push_frame(preprocessed_frame)

                except ValueError as e:
                    logger.error(f"Audio preprocessing failed: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error pushing audio frames: {e}")
        finally:
            vad_stream.end_input()
            logger.info(f"Audio stream ended for {participant_identity}")

    async def process_vad_events():
        """Process VAD events and transcribe speech segments"""
        speech_start_time = None
        auto_flush_timer = None

        async def auto_flush_on_timeout():
            """Auto-flush VAD buffer after 110 seconds to prevent overflow"""
            try:
                await asyncio.sleep(110.0)  # Wait 110 seconds (before 120s limit)
                logger.info(f"⏰ Auto-flush timer expired for {participant_identity} - injecting silence")
                _flush_vad_stream_sync(vad_stream, participant_identity)
            except asyncio.CancelledError:
                logger.debug(f"Auto-flush timer cancelled for {participant_identity}")

        try:
            async for event in vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    logger.info(f"🎙️  Speech started: {participant_identity}")

                    # Start auto-flush timer to prevent buffer overflow
                    speech_start_time = datetime.now()
                    if auto_flush_timer and not auto_flush_timer.done():
                        auto_flush_timer.cancel()
                    auto_flush_timer = asyncio.create_task(auto_flush_on_timeout())

                elif event.type == VADEventType.END_OF_SPEECH:
                    logger.info(f"✋ Speech ended: {participant_identity}, processing {len(event.frames)} frames")

                    # Cancel auto-flush timer (natural END_OF_SPEECH occurred)
                    speech_start_time = None
                    if auto_flush_timer and not auto_flush_timer.done():
                        auto_flush_timer.cancel()
                        auto_flush_timer = None

                    if event.frames:
                        # Capture speech end time NOW (before async transcription)
                        speech_end_time = datetime.now()

                        task = asyncio.create_task(
                            transcribe_and_publish(
                                event.frames,
                                participant_identity,
                                room,
                                speech_end_time
                            )
                        )
                        transcription_tasks.append(task)
                        transcription_tasks[:] = [t for t in transcription_tasks if not t.done()]

        except asyncio.CancelledError:
            logger.info(f"VAD processing cancelled for {participant_identity}")
        except Exception as e:
            logger.error(f"Error processing VAD events: {e}", exc_info=True)
        finally:
            # Cleanup: cancel auto-flush timer
            if auto_flush_timer and not auto_flush_timer.done():
                auto_flush_timer.cancel()

            await vad_stream.aclose()
            if transcription_tasks:
                logger.info(f"Waiting for {len(transcription_tasks)} pending transcriptions...")
                await asyncio.gather(*transcription_tasks, return_exceptions=True)

    await asyncio.gather(
        push_audio_frames(),
        process_vad_events(),
        return_exceptions=True
    )

    # Cleanup: Remove VAD stream from tracking
    if participant_identity in participant_vad_streams:
        del participant_vad_streams[participant_identity]
        logger.debug(f"Cleaned up VAD stream for {participant_identity}")


def _flush_vad_stream_sync(vad_stream, participant_identity: str):
    """Flush VAD by injecting silence to trigger natural END_OF_SPEECH"""
    try:
        # Create 1.5 seconds of silence at 16kHz to trigger VAD's natural silence detection
        silence_samples = int(16000 * 1.5)
        silence_audio = np.zeros(silence_samples, dtype=np.int16)

        # Push silence frame to VAD - this triggers natural END_OF_SPEECH event
        silence_frame = rtc.AudioFrame(
            data=silence_audio.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=silence_samples
        )
        vad_stream.push_frame(silence_frame)
        logger.info(f"💨 Injected silence to flush VAD (mic OFF): {participant_identity}")
    except Exception as e:
        logger.error(f"Error injecting silence for VAD flush: {e}")


def on_data_received(data_packet: rtc.DataPacket, room: Room):
    """Handle incoming data messages (model switch commands)."""
    try:
        sender_id = data_packet.participant.identity if data_packet.participant else 'unknown'
        logger.info(f"📨 Data packet from {sender_id}, size: {len(data_packet.data)} bytes")

        message = json.loads(data_packet.data.decode('utf-8'))
        msg_type = message.get("type")

        if msg_type == "model_switch":
            data = message.get("data", {})
            new_model = data.get("model")
            room_name = data.get("room_name")

            if new_model and room_name:
                previous_model = room_transcription_engines.get(room_name, config.TRANSCRIPTION_ENGINE)
                room_transcription_engines[room_name] = new_model

                logger.info(f"🔄 Model switched: {previous_model} → {new_model} (room: {room_name})")

                if previous_model == "hamza" and new_model != "hamza":
                    asyncio.create_task(_close_hamza_on_switch(room_name))
                    # Send mlx_whisper connected status
                    asyncio.create_task(_send_engine_status(room, "mlx_whisper", "connected"))

                if new_model == "hamza":
                    asyncio.create_task(_init_hamza_client(room))
                elif new_model == "mlx_whisper":
                    asyncio.create_task(_send_engine_status(room, "mlx_whisper", "connected"))

        elif msg_type == "flush_audio":
            logger.info(f"🔄 Received flush audio signal from {sender_id} - mic turned OFF")
            # Manually flush VAD buffer to transcribe accumulated audio
            if sender_id in participant_vad_streams:
                vad_stream = participant_vad_streams[sender_id]
                # Call synchronous flush method directly
                _flush_vad_stream_sync(vad_stream, sender_id)
            else:
                logger.warning(f"No VAD stream found for {sender_id} to flush")

    except json.JSONDecodeError as e:
        logger.warning(f"Non-JSON data packet: {e}")
    except Exception as e:
        logger.error(f"Error handling data packet: {e}", exc_info=True)


async def _send_engine_status(room: Room, engine: str, status: str, error: str = None):
    """Send engine status update to frontend via LiveKit data channel."""
    try:
        from datetime import datetime, timezone
        status_message = {
            "type": "engine_status",
            "data": {
                "engine": engine,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }
        if error:
            status_message["data"]["error"] = error

        await room.local_participant.publish_data(
            json.dumps(status_message).encode('utf-8'),
            reliable=True
        )
        logger.debug(f"📤 Sent engine status: {engine} - {status}")
    except Exception as e:
        logger.error(f"Failed to send engine status: {e}")


async def _init_hamza_client(room: Room = None):
    """Initialize Hamza client in background."""
    try:
        if room:
            await _send_engine_status(room, "hamza", "connecting")

        client = await get_hamza_client()
        if client and client.is_connected:
            logger.info("✅ Hamza WebSocket ready")
            if room:
                await _send_engine_status(room, "hamza", "connected")
        else:
            logger.error("❌ Failed to initialize Hamza")
            if room:
                await _send_engine_status(room, "hamza", "error", "Connection failed")
    except Exception as e:
        logger.error(f"❌ Hamza initialization failed: {e}")
        if room:
            await _send_engine_status(room, "hamza", "error", str(e))


async def _close_hamza_on_switch(room_name: str):
    """Close Hamza client when switching away."""
    try:
        hamza_rooms = [r for r, e in room_transcription_engines.items() if e == "hamza"]
        if not hamza_rooms:
            await close_hamza_client()
            logger.info("✅ Hamza WebSocket closed")
    except Exception as e:
        logger.error(f"Error closing Hamza: {e}")


async def check_nudges_async(context: list, existing_nudges: list, room: rtc.Room, participant: str):
    """Check for nudges using existing llm_service (non-blocking)."""
    try:
        from api import llm_service

        logger.info(f"🔍 Checking nudges for room {room.name} with {len(context)} transcripts")

        # Call existing classify_transcripts function
        nudges = llm_service.classify_transcripts(
            transcripts=context,  # Last 10 transcripts
            api_key=config.OPENAI_API_KEY,
            topic=None,  # No topic needed for transcript-level
            existing_nudges=existing_nudges  # For deduplication
        )

        # If LLM returned new nudges, publish and track them
        if nudges and len(nudges) > 0:
            for nudge in nudges:
                # Add to history
                generated_nudges[room.name].append(nudge)

                # Publish to frontend via LiveKit data channel
                message = {
                    "type": "nudge",
                    "data": nudge
                }
                await room.local_participant.publish_data(
                    payload=json.dumps(message).encode('utf-8'),
                    reliable=True,
                    topic="nudges"
                )
                logger.info(f"📌 Published nudge: {nudge['title']}")
        else:
            logger.debug(f"No new nudges generated (context: {len(context)} transcripts)")

    except Exception as e:
        logger.error(f"Error checking nudges: {e}", exc_info=True)


async def transcribe_and_publish(
    audio_frames: list,
    participant_identity: str,
    room: rtc.Room,
    speech_end_time: datetime
):
    """Transcribe speech segment and publish to room.

    Args:
        audio_frames: Audio frames to transcribe
        participant_identity: Speaker identity
        room: LiveKit room
        speech_end_time: When speech actually ended (not when transcription completes)
    """
    if not audio_frames:
        logger.debug("No audio frames, skipping")
        return

    try:
        # Convert frames to numpy array
        audio_data_source = np.concatenate([
            np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            for frame in audio_frames
        ])

        source_sample_rate = audio_frames[0].sample_rate if audio_frames else 16000
        target_sample_rate = 16000

        duration_seconds = len(audio_data_source) / source_sample_rate

        if duration_seconds < 0.2:
            logger.debug(f"Audio too short ({duration_seconds:.2f}s), skipping")
            return

        logger.info(f"Audio segment: {len(audio_data_source)} samples ({duration_seconds:.2f}s)")

        # Validate audio quality
        if config.ENABLE_AUDIO_VALIDATION:
            is_valid, reason = audio_preprocessing.validate_audio_quality(
                audio_data_source,
                sample_rate=source_sample_rate,
                silence_threshold_db=config.AUDIO_VALIDATION_RMS_THRESHOLD_DB
            )
            if not is_valid:
                logger.warning(f"⚠️ Audio rejected: {reason}")
                return

        # Resample to 16kHz
        if source_sample_rate != target_sample_rate:
            if source_sample_rate == 48000:
                audio_data = resample_poly(audio_data_source, 1, 3)
            else:
                num_samples = int(len(audio_data_source) * target_sample_rate / source_sample_rate)
                audio_data = scipy_signal.resample(audio_data_source, num_samples)
        else:
            audio_data = audio_data_source

        # Get current engine from in-memory state (or default from .env)
        current_engine = room_transcription_engines.get(room.name, config.TRANSCRIPTION_ENGINE)
        room_transcription_engines[room.name] = current_engine
        logger.info(f"Using transcription engine: {current_engine}")

        # Track if we published final via callback (to avoid duplicate)
        published_final_via_callback = False

        # Create callback for real-time partial transcripts (Hamza only)
        async def publish_partial_transcript(text: str, is_final: bool = False):
            """Publish partial transcript to UI in real-time and add to segment manager"""
            nonlocal published_final_via_callback
            try:
                # Use speech end time (when person stopped talking), not current time
                timestamp = speech_end_time.strftime("%H:%M:%S")
                partial_transcript = {
                    "timestamp": timestamp,
                    "speaker": participant_identity,
                    "text": text,
                    "is_final": is_final,  # Use parameter
                    "engine": "hamza"
                }
                message = {"type": "transcript", "data": partial_transcript}
                payload = json.dumps(message).encode('utf-8')

                await room.local_participant.publish_data(
                    payload=payload,
                    reliable=True,
                    topic="transcription"
                )
                logger.debug(f"Published partial transcript: {text}...")

                # Add to segment manager (real-time segmentation)
                if room.name in segment_managers:
                    manager = segment_managers[room.name]
                    manager.add_transcript(
                        timestamp=timestamp,
                        text=text,
                        speaker=participant_identity
                    )
                    logger.debug(f"✅ Partial transcript added to SegmentManager")

                # Track if we published final
                if is_final:
                    published_final_via_callback = True
            except Exception as e:
                logger.error(f"Error publishing partial transcript: {e}")

        # Transcribe
        if current_engine == "hamza":
            result = await transcribe_audio_hamza(
                audio_data,
                sample_rate=target_sample_rate,
                is_final=True,
                on_partial_result=publish_partial_transcript
            )

            if result.get("error"):
                logger.error(f"❌ Hamza error: {result['error']}")
                return

            if "metrics" in result:
                m = result["metrics"]
                logger.info(f"📊 Hamza: latency={m.get('latency_ms', 0):.0f}ms")
        else:
            result = transcribe_audio(audio_data, sample_rate=target_sample_rate, is_final=True)
            result["engine"] = "mlx_whisper"

        text = result.get("text", "").strip()
        engine_used = result.get('engine', current_engine)
        logger.info(f"Transcription completed using: {engine_used}")

        if not text:
            logger.debug("Empty transcription, skipping")
            return

        # Hallucination detection
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                w = word.lower().strip('.,!?')
                word_counts[w] = word_counts.get(w, 0) + 1

            max_count = max(word_counts.values())
            if max_count > len(words) * 0.35:
                most_repeated = max(word_counts, key=word_counts.get)
                logger.warning(f"⚠️ Hallucination: '{most_repeated}' repeated {max_count}/{len(words)}")
                return

        # Check for known hallucination phrases
        HALLUCINATION_PHRASES = [
            "thank you for watching", "thanks for watching",
            "subscribe", "like and subscribe",
            "see you next time", "bye bye bye",
        ]
        text_lower = text.lower()
        for phrase in HALLUCINATION_PHRASES:
            if phrase in text_lower:
                logger.warning(f"⚠️ Hallucination: contains '{phrase}'")
                return

        # Use speech end time (when person stopped talking), not current time
        timestamp = speech_end_time.strftime("%H:%M:%S")
        logger.info(f"[{timestamp}] {participant_identity}: {text}")

        # Publish transcript (skip for Hamza - callback handles all publishes)
        if current_engine == "hamza":
            logger.info("Skipping normal publish for Hamza (callback handles it)")
        else:
            transcript = {
                "timestamp": timestamp,
                "speaker": participant_identity,
                "text": text,
                "is_final": True,
                "engine": engine_used
            }

            message = {"type": "transcript", "data": transcript}
            payload = json.dumps(message).encode('utf-8')

            await room.local_participant.publish_data(
                payload=payload,
                reliable=True,
                topic="transcription"
            )
            logger.info("Published transcript to room")

        # Add to segment manager (skip for Hamza - callback handles it)
        if current_engine == "hamza":
            logger.debug("Skipping segment manager add for Hamza (callback handles it)")
        else:
            if room.name in segment_managers:
                manager = segment_managers[room.name]
                manager.add_transcript(
                    timestamp=timestamp,
                    text=text,
                    speaker=participant_identity
                )
                logger.info("✅ Transcript added to SegmentManager")

        # Initialize nudge buffers for this room if needed
        if room.name not in nudge_context_buffer:
            nudge_context_buffer[room.name] = []
            generated_nudges[room.name] = []

        # Add transcript to nudge context buffer
        nudge_context_buffer[room.name].append(text)
        if len(nudge_context_buffer[room.name]) > config.NUDGE_CONTEXT_SIZE:
            nudge_context_buffer[room.name].pop(0)

        # Check for nudges (non-blocking, every transcript)
        asyncio.create_task(
            check_nudges_async(
                context=nudge_context_buffer[room.name].copy(),
                existing_nudges=generated_nudges[room.name].copy(),
                room=room,
                participant=participant_identity
            )
        )

    except Exception as e:
        logger.error(f"Error in transcribe_and_publish: {e}", exc_info=True)


def prewarm(proc: JobProcess):
    """Prewarm models before agent starts."""
    logger.info("🔥 Prewarming models...")

    logger.info("🎚️ Audio preprocessing config:")
    logger.info(f"   - High-pass filter: {'enabled' if config.ENABLE_HIGHPASS_FILTER else 'disabled'}")
    logger.info(f"   - Normalization: {'enabled' if config.ENABLE_NORMALIZATION else 'disabled'}")
    logger.info(f"   - Audio validation: {'enabled' if config.ENABLE_AUDIO_VALIDATION else 'disabled'}")
    logger.info(f"🎙️ Default engine: {config.TRANSCRIPTION_ENGINE}")

    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.35,
        min_speech_duration=0.25,
        min_silence_duration=1.2,
        prefix_padding_duration=1.0,
        max_buffered_speech=120.0,  # 2 minutes - auto-segments long speeches, prevents buffer overflow
        sample_rate=16000
    )
    logger.info("✅ VAD loaded")

    logger.info("🔧 Creating LLM classifier client...")
    classifier_client = LLMClassifierClient()
    classifier_client.start()
    logger.info("✅ LLM classifier started")
    proc.userdata["classifier_client"] = classifier_client


async def entrypoint(ctx: JobContext):
    """Agent entrypoint."""
    logger.info("=" * 80)
    logger.info("🚀 AGENT ENTRYPOINT CALLED!")
    logger.info(f"   Room: {ctx.room.name if hasattr(ctx, 'room') else 'pending'}")
    logger.info("=" * 80)

    await ctx.connect()
    room = ctx.room

    await asyncio.sleep(1)

    # Check for duplicate agents
    for participant in room.remote_participants.values():
        if participant.identity.startswith("agent-"):
            logger.warning(f"⚠️ DUPLICATE AGENT! Exiting...")
            return

    logger.info(f"✅ Connected to room: {room.name}")

    vad = ctx.proc.userdata["vad"]
    classifier_client = ctx.proc.userdata["classifier_client"]

    # Load Whisper if needed
    if config.TRANSCRIPTION_ENGINE == "mlx_whisper":
        logger.info(f"📥 Loading Whisper model: {config.WHISPER_MODEL}")
        load_whisper_model(config.WHISPER_MODEL)

    # Create segment manager
    segment_managers[room.name] = SegmentManager(
        meeting_id=room.name,
        room=room,
        classifier_client=classifier_client
    )
    logger.info("✅ SegmentManager created")

    # Set engine
    room_transcription_engines[room.name] = config.TRANSCRIPTION_ENGINE
    logger.info(f"🎙️ Engine: {room_transcription_engines[room.name]}")

    # Pre-initialize Hamza if needed
    if config.TRANSCRIPTION_ENGINE == "hamza":
        logger.info("🔧 Pre-initializing Hamza...")
        await _send_engine_status(room, "hamza", "connecting")
        try:
            client = await get_hamza_client()
            if client and client.is_connected:
                logger.info("✅ Hamza ready")
                await _send_engine_status(room, "hamza", "connected")
            else:
                logger.error("❌ Hamza initialization failed")
                await _send_engine_status(room, "hamza", "error", "Connection failed")
        except Exception as e:
            logger.error(f"❌ Hamza error: {e}")
            await _send_engine_status(room, "hamza", "error", str(e))
    else:
        # Send mlx_whisper connected status
        await _send_engine_status(room, "mlx_whisper", "connected")

    participant_tasks = {}
    shutdown_event = asyncio.Event()

    try:
        @room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                if participant.identity in participant_tasks:
                    if not participant_tasks[participant.identity].done():
                        return

                logger.info(f"🎵 Audio track from {participant.identity}")
                task = asyncio.create_task(
                    process_audio_with_vad(track, participant.identity, room, vad)
                )
                participant_tasks[participant.identity] = task

        @room.on("participant_connected")
        def on_participant_connected(participant):
            logger.info(f"👤 Participant connected: {participant.identity}")
            # Send current engine status to newly connected participant
            engine = config.TRANSCRIPTION_ENGINE
            asyncio.create_task(_send_engine_status(room, engine, "connected"))

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant):
            logger.info(f"👋 Participant disconnected: {participant.identity}")
            if participant.identity in participant_tasks:
                participant_tasks[participant.identity].cancel()
                del participant_tasks[participant.identity]

        @room.on("disconnected")
        def on_disconnected():
            logger.info("🔌 Room disconnected")
            shutdown_event.set()

        # Create closure to capture room for data handler
        def on_data_received_with_room(data_packet: rtc.DataPacket):
            on_data_received(data_packet, room)

        room.on("data_received", on_data_received_with_room)
        logger.info("✅ Agent ready!")

        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info("🧹 Cleaning up...")
        
        for identity, task in list(participant_tasks.items()):
            if not task.done():
                task.cancel()

        if participant_tasks:
            await asyncio.gather(*participant_tasks.values(), return_exceptions=True)

        if room.name in segment_managers:
            await segment_managers[room.name].cleanup()
            del segment_managers[room.name]

        # Cleanup in-memory model state
        if room.name in room_transcription_engines:
            del room_transcription_engines[room.name]
            logger.debug(f"Removed in-memory model config for {room.name}")

        if len(room_transcription_engines) == 0:
            await close_hamza_client()

        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    from livekit.agents import WorkerOptions, WorkerType

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            agent_name="production-agent",
        )
    )