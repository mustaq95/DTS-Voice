"""
Production-Ready Hamza WebSocket STT Client

Features:
- True streaming (send audio as it arrives)
- Partial results for real-time display
- Auto-reconnect with exponential backoff
- Backpressure handling
- Metrics and monitoring
- Backward compatible batch API
"""
import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
import numpy as np
import websockets

# Import config - adjust path as needed for your project
try:
    from . import config
except ImportError:
    # Fallback for standalone testing
    class config:
        HAMZA_WS_URL = "wss://beta.govgpt.gov.ae/llm/hamsa/ws"
        HAMZA_API_KEY = ""
        HAMZA_SILENCE_TIMEOUT_MS = 3000
        HAMZA_MIN_SILENCE_MS = 300
        HAMZA_MIN_SPEECH_MS = 150
        HAMZA_VAD_THRESHOLD = 0.2
        HAMZA_EOS_ENABLED = True
        HAMZA_EOS_THRESHOLD = 0.5

logger = logging.getLogger("hamza-stt")


# ============================================
# Data Classes
# ============================================

@dataclass
class TranscriptionResult:
    """Structured transcription result"""
    text: str
    is_final: bool
    is_partial: bool = False
    confidence: float = 0.0
    latency_ms: float = 0.0
    engine: str = "hamza"


@dataclass
class StreamingMetrics:
    """Track streaming performance"""
    chunks_sent: int = 0
    bytes_sent: int = 0
    results_received: int = 0
    partial_results: int = 0
    final_results: int = 0
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)
    first_result_time: Optional[float] = None

    @property
    def avg_latency_ms(self) -> float:
        if self.results_received == 0:
            return 0.0
        return self.total_latency_ms / self.results_received

    @property
    def time_to_first_result_ms(self) -> float:
        if self.first_result_time is None:
            return 0.0
        return (self.first_result_time - self.start_time) * 1000

    @property
    def duration_s(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks_sent": self.chunks_sent,
            "bytes_sent": self.bytes_sent,
            "results_received": self.results_received,
            "partial_results": self.partial_results,
            "final_results": self.final_results,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "time_to_first_result_ms": round(self.time_to_first_result_ms, 1),
            "duration_s": round(self.duration_s, 2)
        }


# ============================================
# Production Streaming Client
# ============================================

class HamzaStreamingClient:
    """
    Production-ready streaming STT client.

    Key Features:
    1. True streaming - sends audio chunks as they arrive
    2. Partial results - get text before speech ends
    3. Non-blocking - uses separate send/receive tasks
    4. Auto-reconnect - handles connection drops gracefully
    5. Backpressure - doesn't overwhelm the server
    6. Metrics - track performance for monitoring

    Usage:
        # Real-time streaming
        client = HamzaStreamingClient(ws_url, api_key)
        await client.connect()
        await client.start_streaming()

        for audio_chunk in audio_frames:
            await client.send_audio_chunk(audio_chunk)

        final_text = await client.stop_streaming()

        # Or batch mode (backward compatible)
        result = await client.transcribe_batch(audio_data)
    """

    def __init__(
        self,
        ws_url: str = None,
        api_key: str = None,
        on_partial_result: Optional[Callable[[str], Awaitable[None]]] = None,
        on_final_result: Optional[Callable[[str], Awaitable[None]]] = None,
        max_queue_size: int = 100,
    ):
        """
        Initialize Hamza streaming client.

        Args:
            ws_url: WebSocket URL (defaults to config.HAMZA_WS_URL)
            api_key: API key (defaults to config.HAMZA_API_KEY)
            on_partial_result: Async callback for partial transcripts
            on_final_result: Async callback for final transcripts
            max_queue_size: Max audio chunks to queue before dropping
        """
        self.ws_url = ws_url or config.HAMZA_WS_URL
        self.api_key = api_key or config.HAMZA_API_KEY
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False

        # Callbacks for real-time results
        self.on_partial_result = on_partial_result
        self.on_final_result = on_final_result

        # Streaming state
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._result_queue: asyncio.Queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._is_streaming = False
        self._lock = asyncio.Lock()

        # Metrics
        self.metrics = StreamingMetrics()

        # Connection management
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        self._reconnect_delay = 1.0

        # Latest result tracking
        self._latest_text = ""
        self._all_partials = []  # Collect all partial fragments for concatenation

    async def connect(self) -> bool:
        """
        Connect to Hamza WebSocket and perform handshake.

        Returns:
            bool: True if connected successfully
        """
        try:
            logger.info(f"🔌 Connecting to Hamza: {self.ws_url}")

            self.ws = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    ping_interval=5,
                    ping_timeout=20,
                    max_size=10 * 1024 * 1024,  # 10MB max message
                    compression=None,  # Disable compression for lower latency
                ),
                timeout=10.0
            )

            # Log configuration
            logger.info(f"🔧 Hamza config:")
            logger.info(f"   min_silence_ms: {config.HAMZA_MIN_SILENCE_MS}")
            logger.info(f"   min_speech_ms: {config.HAMZA_MIN_SPEECH_MS}")
            logger.info(f"   vad_threshold: {config.HAMZA_VAD_THRESHOLD}")
            logger.info(f"   eos_enabled: {config.HAMZA_EOS_ENABLED}")

            # Send handshake
            handshake = {
                "type": "handshake",
                "api_key": self.api_key,
                "options": {
                    "silence_timeout": config.HAMZA_SILENCE_TIMEOUT_MS,
                    "sample_rate": 16000,
                    "min_silence_duration_ms": config.HAMZA_MIN_SILENCE_MS,
                    "min_speech_ms": config.HAMZA_MIN_SPEECH_MS,
                    "client_logging": False,
                    "vad_threshold": config.HAMZA_VAD_THRESHOLD,
                    "eos_enabled": config.HAMZA_EOS_ENABLED,
                    "eos_threshold": config.HAMZA_EOS_THRESHOLD,
                    "audio_type": "PCM",
                }
            }
            await self.ws.send(json.dumps(handshake))
            logger.info("🤝 Handshake sent, waiting for ACK...")

            # Wait for ACK
            msg = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(msg)

            if data.get("type") == "handshake_ack":
                logger.info("✅ Hamza connected and ready")
                self.is_connected = True
                self._reconnect_attempts = 0
                return True
            else:
                logger.error(f"❌ Handshake failed: {data}")
                return False

        except asyncio.TimeoutError:
            logger.error("❌ Connection timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    async def _auto_reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Returns:
            bool: True if reconnected successfully
        """
        while self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))

            logger.warning(
                f"🔄 Reconnecting (attempt {self._reconnect_attempts}/"
                f"{self._max_reconnect_attempts}) in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

            if await self.connect():
                logger.info("✅ Reconnected successfully")
                return True

        logger.error("❌ Max reconnection attempts reached")
        return False

    @staticmethod
    def float_to_pcm16(audio_data: np.ndarray) -> bytes:
        """
        Convert float32 audio to PCM16 bytes.

        Args:
            audio_data: Float32 audio array (-1.0 to 1.0)

        Returns:
            bytes: PCM16 little-endian bytes
        """
        audio_data = np.clip(audio_data, -1.0, 1.0)
        return (audio_data * 32767).astype(np.int16).tobytes()

    async def start_streaming(self):
        """
        Start the streaming send/receive tasks.
        Call this before sending audio chunks.
        """
        if self._is_streaming:
            logger.debug("Already streaming, skipping start")
            return

        if not self.is_connected:
            logger.warning("Not connected, attempting to connect...")
            if not await self.connect():
                raise RuntimeError("Failed to connect to Hamza")

        self._is_streaming = True
        self.metrics = StreamingMetrics()
        self._latest_text = ""
        self._all_partials = []  # Reset partial collection

        # Clear queues
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Start background tasks
        self._send_task = asyncio.create_task(self._send_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())

        logger.info("🎙️ Streaming started")

    async def stop_streaming(self) -> str:
        """
        Stop streaming and get final result.

        Returns:
            str: Final transcription text
        """
        if not self._is_streaming:
            return self._latest_text

        logger.debug("Stopping streaming...")

        # Send silence to trigger final result
        await self._send_end_of_speech_signal()

        # Wait a bit for final results
        await asyncio.sleep(0.3)

        # Stop streaming flag
        self._is_streaming = False

        # Collect any remaining results
        final_text = self._latest_text
        wait_start = time.time()
        max_wait = 2.0  # Max 2 seconds

        while (time.time() - wait_start) < max_wait:
            try:
                result = await asyncio.wait_for(self._result_queue.get(), timeout=0.1)
                if result.text:
                    final_text = result.text
                    if result.is_final:
                        break
            except asyncio.TimeoutError:
                break

        # Cancel tasks gracefully
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            try:
                await asyncio.wait_for(self._send_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await asyncio.wait_for(self._receive_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        self._send_task = None
        self._receive_task = None

        # Log metrics
        logger.info(
            f"📊 Stream complete: {self.metrics.chunks_sent} chunks, "
            f"{self.metrics.bytes_sent / 1024:.1f}KB sent, "
            f"{self.metrics.results_received} results, "
            f"avg latency: {self.metrics.avg_latency_ms:.0f}ms, "
            f"TTFR: {self.metrics.time_to_first_result_ms:.0f}ms"
        )

        # Build complete transcript from all partials
        if self._all_partials:
            complete_transcript = " ".join(self._all_partials)
            logger.info(f"✅ Complete transcript ({len(self._all_partials)} fragments): {complete_transcript[:100]}...")
            return complete_transcript
        else:
            # Fallback to final_text if no partials collected
            return final_text

    async def send_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Send audio chunk for streaming transcription.
        Call this continuously as audio arrives (every 20-50ms).

        Non-blocking - queues the chunk for background sending.

        Args:
            audio_chunk: Float32 audio array
        """
        if not self._is_streaming:
            await self.start_streaming()

        pcm_bytes = self.float_to_pcm16(audio_chunk)
        timestamp = time.time()

        try:
            # Non-blocking put
            self._send_queue.put_nowait((pcm_bytes, timestamp))
        except asyncio.QueueFull:
            # Drop oldest chunk if queue is full (backpressure)
            try:
                self._send_queue.get_nowait()
                logger.warning("⚠️ Send queue full, dropping oldest chunk")
            except asyncio.QueueEmpty:
                pass

            try:
                self._send_queue.put_nowait((pcm_bytes, timestamp))
            except asyncio.QueueFull:
                logger.error("❌ Failed to queue chunk even after dropping")

    async def _send_loop(self):
        """Background task: Send audio chunks to server"""
        try:
            while self._is_streaming or not self._send_queue.empty():
                try:
                    # Get chunk with timeout
                    pcm_bytes, timestamp = await asyncio.wait_for(
                        self._send_queue.get(),
                        timeout=0.05
                    )

                    if self.ws and self.is_connected:
                        await self.ws.send(pcm_bytes)
                        self.metrics.chunks_sent += 1
                        self.metrics.bytes_sent += len(pcm_bytes)
                    else:
                        logger.warning("⚠️ WebSocket not available, chunk dropped")

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"❌ Connection lost in send loop: {e}")
                    self.is_connected = False
                    if self._is_streaming and not await self._auto_reconnect():
                        break
                except Exception as e:
                    logger.error(f"❌ Send error: {e}")
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.debug("Send loop cancelled")
        except Exception as e:
            logger.error(f"❌ Send loop error: {e}", exc_info=True)

    async def _receive_loop(self):
        """Background task: Receive results from server"""
        try:
            while self._is_streaming:
                try:
                    if not self.ws or not self.is_connected:
                        await asyncio.sleep(0.05)
                        continue

                    msg = await asyncio.wait_for(self.ws.recv(), timeout=0.05)
                    receive_time = time.time()
                    data = json.loads(msg)

                    if data.get("type") == "transcription":
                        text = data.get("data", {}).get("transcription", "").strip()
                        is_final = data.get("data", {}).get("is_final", False)

                        # Some APIs use different field names
                        if not text:
                            text = data.get("data", {}).get("text", "").strip()

                        if text:
                            # Track first result time
                            if self.metrics.first_result_time is None:
                                self.metrics.first_result_time = receive_time

                            # Calculate latency from stream start
                            latency_ms = (receive_time - self.metrics.start_time) * 1000

                            result = TranscriptionResult(
                                text=text,
                                is_final=is_final,
                                is_partial=not is_final,
                                latency_ms=latency_ms
                            )

                            # Update metrics
                            self.metrics.results_received += 1
                            self.metrics.total_latency_ms += latency_ms
                            if is_final:
                                self.metrics.final_results += 1
                            else:
                                self.metrics.partial_results += 1

                            # Update latest text
                            self._latest_text = text

                            # Collect all partials for complete transcript
                            if not is_final:
                                self._all_partials.append(text)

                            # Put in result queue
                            try:
                                self._result_queue.put_nowait(result)
                            except asyncio.QueueFull:
                                pass

                            # Call callbacks
                            try:
                                if is_final and self.on_final_result:
                                    await self.on_final_result(text)
                                elif not is_final and self.on_partial_result:
                                    await self.on_partial_result(text)
                            except Exception as e:
                                logger.error(f"❌ Callback error: {e}")

                            logger.debug(
                                f"{'✅' if is_final else '📝'} "
                                f"{'Final' if is_final else 'Partial'}: "
                                f"'{text[:50]}{'...' if len(text) > 50 else ''}' "
                                f"({latency_ms:.0f}ms)"
                            )

                    elif data.get("type") == "error":
                        logger.error(f"❌ Hamza server error: {data.get('error')}")

                    elif data.get("type") == "log":
                        logger.debug(f"📋 Hamza log: {data.get('data')}")

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"❌ Connection lost in receive loop: {e}")
                    self.is_connected = False
                    if self._is_streaming and not await self._auto_reconnect():
                        break
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Invalid JSON from server: {e}")
                except Exception as e:
                    logger.error(f"❌ Receive error: {e}")
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"❌ Receive loop error: {e}", exc_info=True)

    async def _send_end_of_speech_signal(self):
        """Send silence to trigger end-of-speech detection"""
        if not self.ws or not self.is_connected:
            return

        # Calculate silence duration (longer than min_silence_ms)
        silence_duration_ms = max(600, config.HAMZA_MIN_SILENCE_MS + 300)
        silence_samples = int(16000 * silence_duration_ms / 1000)
        silence = np.zeros(silence_samples, dtype=np.float32)
        pcm_bytes = self.float_to_pcm16(silence)

        logger.debug(f"📤 Sending {silence_duration_ms}ms silence for EOS")

        # Send silence in chunks
        chunk_size = 1024
        try:
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i + chunk_size]
                await self.ws.send(chunk)
                await asyncio.sleep(0.005)
        except Exception as e:
            logger.warning(f"⚠️ Error sending EOS silence: {e}")

    async def transcribe_batch(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Batch transcription API (backward compatible).
        Internally uses streaming for better performance.

        Args:
            audio_data: Float32 audio array
            sample_rate: Sample rate (default 16kHz)

        Returns:
            dict: Transcription result with text, metrics, etc.
        """
        async with self._lock:
            try:
                if not self.is_connected:
                    if not await self.connect():
                        return {
                            "text": "",
                            "error": "Connection failed",
                            "engine": "hamza"
                        }

                # Log input audio stats
                rms = np.sqrt(np.mean(audio_data**2))
                duration_ms = len(audio_data) / 16  # samples / 16 = ms at 16kHz
                logger.info(
                    f"🎙️ Batch transcribe: {len(audio_data)} samples "
                    f"({duration_ms:.0f}ms), RMS={rms:.4f}"
                )

                # Start streaming
                await self.start_streaming()

                # Send audio in chunks (simulate real-time for best results)
                chunk_size = 512  # 32ms at 16kHz
                chunk_delay = 0.015  # 15ms between chunks (slightly faster than real-time)

                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    await self.send_audio_chunk(chunk)
                    await asyncio.sleep(chunk_delay)

                # Stop and get result
                final_text = await self.stop_streaming()

                return {
                    "text": final_text or "",
                    "is_final": True,
                    "language": "ar",
                    "engine": "hamza",
                    "metrics": self.metrics.to_dict()
                }

            except Exception as e:
                logger.error(f"❌ Batch transcription error: {e}", exc_info=True)
                return {
                    "text": "",
                    "error": str(e),
                    "engine": "hamza"
                }

    async def close(self):
        """Close connection and cleanup all resources"""
        logger.info("🧹 Closing Hamza client...")

        self._is_streaming = False

        # Cancel tasks
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            try:
                await asyncio.wait_for(self._send_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await asyncio.wait_for(self._receive_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Close WebSocket
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.debug(f"WebSocket close error (expected): {e}")

        self.ws = None
        self.is_connected = False
        self._send_task = None
        self._receive_task = None

        logger.info("✅ Hamza client closed")


# ============================================
# Singleton API (Backward Compatible)
# ============================================

_hamza_client: Optional[HamzaStreamingClient] = None
_client_lock = asyncio.Lock()


async def get_hamza_client() -> Optional[HamzaStreamingClient]:
    """
    Get or create Hamza client singleton.

    Returns:
        HamzaStreamingClient or None if connection fails
    """
    global _hamza_client

    async with _client_lock:
        if _hamza_client is None or not _hamza_client.is_connected:
            logger.info("🔧 Initializing Hamza streaming client...")

            _hamza_client = HamzaStreamingClient(
                ws_url=config.HAMZA_WS_URL,
                api_key=config.HAMZA_API_KEY
            )

            if not await _hamza_client.connect():
                logger.error("❌ Failed to initialize Hamza client")
                _hamza_client = None
                return None

        return _hamza_client


async def transcribe_audio_hamza(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    is_final: bool = True
) -> Dict[str, Any]:
    """
    Public transcription API (backward compatible).

    Args:
        audio_data: Float32 audio array (-1.0 to 1.0)
        sample_rate: Sample rate (default 16kHz)
        is_final: Whether this is final audio (for compatibility)

    Returns:
        dict: {
            "text": str,
            "is_final": bool,
            "language": str,
            "engine": str,
            "metrics": dict (optional),
            "error": str (optional)
        }
    """
    client = await get_hamza_client()

    if client is None:
        return {
            "text": "",
            "error": "Client not available",
            "engine": "hamza"
        }

    return await client.transcribe_batch(audio_data, sample_rate)


async def close_hamza_client():
    """Close Hamza client and release resources"""
    global _hamza_client

    async with _client_lock:
        if _hamza_client:
            await _hamza_client.close()
            _hamza_client = None
            logger.info("✅ Hamza singleton client closed")


# ============================================
# LiveKit Integration Helper
# ============================================

class HamzaLiveKitStreamer:
    """
    Helper for integrating Hamza streaming with LiveKit audio.

    Usage:
        streamer = HamzaLiveKitStreamer(room)
        await streamer.start()

        # For each audio frame from LiveKit:
        await streamer.process_frame(audio_frame)

        # When VAD detects end of speech:
        final_text = await streamer.finalize()
    """

    def __init__(
        self,
        room,  # LiveKit room
        on_partial: Optional[Callable[[str], Awaitable[None]]] = None,
        on_final: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.room = room
        self.client: Optional[HamzaStreamingClient] = None
        self._on_partial = on_partial
        self._on_final = on_final

    async def start(self):
        """Initialize and start streaming"""
        self.client = HamzaStreamingClient(
            on_partial_result=self._handle_partial,
            on_final_result=self._handle_final
        )

        if not await self.client.connect():
            raise RuntimeError("Failed to connect to Hamza")

        await self.client.start_streaming()
        logger.info("✅ Hamza LiveKit streamer ready")

    async def _handle_partial(self, text: str):
        """Handle partial result"""
        if self._on_partial:
            await self._on_partial(text)

        # Publish to room (unreliable for speed)
        await self._publish(text, is_final=False)

    async def _handle_final(self, text: str):
        """Handle final result"""
        if self._on_final:
            await self._on_final(text)

        # Publish to room (reliable for accuracy)
        await self._publish(text, is_final=True)

    async def _publish(self, text: str, is_final: bool):
        """Publish transcript to LiveKit room"""
        import json

        message = {
            "type": "transcript",
            "data": {
                "text": text,
                "is_final": is_final,
                "engine": "hamza"
            }
        }

        try:
            await self.room.local_participant.publish_data(
                payload=json.dumps(message).encode('utf-8'),
                reliable=is_final,
                topic="transcription"
            )
        except Exception as e:
            logger.error(f"❌ Failed to publish transcript: {e}")

    async def process_frame(self, audio_data: np.ndarray):
        """
        Process audio frame from LiveKit.

        Args:
            audio_data: Float32 audio array (already at 16kHz)
        """
        if self.client:
            await self.client.send_audio_chunk(audio_data)

    async def finalize(self) -> str:
        """
        Finalize current utterance and get result.
        Call this when VAD detects end of speech.

        Returns:
            str: Final transcription text
        """
        if not self.client:
            return ""

        final_text = await self.client.stop_streaming()

        # Restart streaming for next utterance
        await self.client.start_streaming()

        return final_text

    async def close(self):
        """Cleanup resources"""
        if self.client:
            await self.client.close()
            self.client = None


# ============================================
# Testing
# ============================================

async def _test_client():
    """Simple test of the client"""
    import numpy as np

    # Create test audio (1 second of silence + tone)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print("Testing Hamza client...")

    # Test batch mode
    result = await transcribe_audio_hamza(audio, sample_rate)
    print(f"Result: {result}")

    # Cleanup
    await close_hamza_client()
    print("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(_test_client())
