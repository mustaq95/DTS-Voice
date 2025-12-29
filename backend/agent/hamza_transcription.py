"""
Near-Realtime Hamza WebSocket STT Client
Continuous streaming architecture matching colleague's implementation
"""
import asyncio
import json
import logging
import numpy as np
import websockets
from typing import Optional, Dict, Any, Callable

try:
    from . import config
except ImportError:
    class config:
        HAMZA_WS_URL = "wss://beta.govgpt.gov.ae/llm/hamsa/ws"
        HAMZA_API_KEY = ""
        HAMZA_SILENCE_TIMEOUT_MS = 3000

logger = logging.getLogger("hamza-stt")


class HamzaClient:
    """Continuous streaming client - mimics colleague's near-realtime architecture"""

    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self._lock = asyncio.Lock()
        self._listener_task: Optional[asyncio.Task] = None
        self._current_callback: Optional[Callable] = None
        self._current_result = {"text": "", "is_final": False}
        self._result_event = asyncio.Event()
        self._last_published_text = ""  # Track what we've already published

    async def connect(self) -> bool:
        """Connect and handshake"""
        try:
            logger.info("🔌 Connecting to Hamza...")

            self.ws = await asyncio.wait_for(
                websockets.connect(
                    config.HAMZA_WS_URL,
                    ping_interval=5,
                    ping_timeout=20,
                ),
                timeout=10.0
            )

            # Colleague's optimized handshake settings
            handshake = {
                "type": "handshake",
                "api_key": config.HAMZA_API_KEY,
                "options": {
                    "silence_timeout": config.HAMZA_SILENCE_TIMEOUT_MS,
                    "sample_rate": 16000,
                    "min_silence_duration_ms": config.HAMZA_MIN_SILENCE_MS,
                    "min_speech_ms": config.HAMZA_MIN_SPEECH_MS,
                    "vad_threshold": config.HAMZA_VAD_THRESHOLD,
                    "eos_enabled": config.HAMZA_EOS_ENABLED,
                    "eos_threshold": config.HAMZA_EOS_THRESHOLD,
                    # "min_silence_duration_ms": 300,  # 0.3s for fast EOS
                    # "min_speech_ms": 100,  # 0.1s for fast detection
                    # "vad_threshold": 0.3,  # Sensitive detection
                    # "eos_enabled": True,
                    # "eos_threshold": 0.6,
                    "audio_type": "PCM",
                    "client_logging": True,
                }
            }
            await self.ws.send(json.dumps(handshake))

            msg = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(msg)

            if data.get("type") == "handshake_ack":
                logger.info("✅ Hamza connected - near-realtime streaming mode")
                self.is_connected = True
                return True

            logger.error(f"❌ Handshake failed: {data}")
            self.is_connected = False
            return False

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.is_connected = False
            return False

    async def _continuous_listener(self):
        """Background listener for continuous partial results"""
        try:
            while self.is_connected and self.ws:
                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                    data = json.loads(msg)

                    if data.get("type") == "transcription":
                        text = data.get("data", {}).get("transcription", "").strip()
                        if text and text != self._last_published_text:
                            self._current_result["text"] = text
                            self._last_published_text = text
                            logger.info(f"📝 Partial result: '{text[:50]}...'")

                            # Call callback immediately for real-time updates
                            if self._current_callback:
                                try:
                                    await self._current_callback(text, is_final=False)
                                except Exception as e:
                                    logger.error(f"Error in partial callback: {e}")

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Listener error: {e}")
                    break

        except Exception as e:
            logger.error(f"Listener crashed: {e}")

    async def transcribe(self, audio_data: np.ndarray, on_partial_result=None) -> Dict[str, Any]:
        """
        Continuous streaming transcription matching colleague's architecture.

        Key differences from old approach:
        - No silence padding (sends raw audio immediately)
        - Continuous chunk streaming
        - Background listener for partial results
        - Minimal latency

        Args:
            audio_data: Audio samples to transcribe (16kHz)
            on_partial_result: Optional async callback(text: str, is_final: bool)

        Returns:
            Dict with 'text', 'is_final', and 'engine' keys
        """
        async with self._lock:
            try:
                # Ensure connected
                if not self.is_connected or not self.ws:
                    if not await self.connect():
                        return {"text": "", "error": "Connection failed", "engine": "hamza"}

                # Set callback PERSISTENTLY (not cleared until next transcription)
                if on_partial_result:
                    self._current_callback = on_partial_result
                self._current_result = {"text": "", "is_final": False}
                self._result_event.clear()
                self._last_published_text = ""  # Reset for new transcription

                # Start background listener if not running
                if self._listener_task is None or self._listener_task.done():
                    self._listener_task = asyncio.create_task(self._continuous_listener())

                # Add 500ms silence padding at both ends for better VAD/EOS detection
                silence_samples = np.zeros(int(16000 * 0.5), dtype=np.float32)
                audio_with_silence = np.concatenate([silence_samples, audio_data, silence_samples])

                # Convert to PCM16
                audio_clipped = np.clip(audio_with_silence, -1.0, 1.0)
                pcm_bytes = (audio_clipped * 32767).astype(np.int16).tobytes()

                duration_s = len(audio_data) / 16000
                logger.info(f"🎙️ Streaming {duration_s:.1f}s audio (+1.0s silence padding)")

                # Stream ALL audio immediately in small chunks (colleague's approach)
                chunk_size = 256  # Same as colleague
                total_chunks = len(pcm_bytes) // chunk_size + (1 if len(pcm_bytes) % chunk_size else 0)

                for i in range(0, len(pcm_bytes), chunk_size):
                    chunk = pcm_bytes[i:i + chunk_size]
                    await self.ws.send(chunk)
                    # Small delay to avoid overwhelming the server
                    await asyncio.sleep(0.001)

                logger.info(f"✅ Sent {total_chunks} chunks")

                # Wait briefly for final result (colleague waits ~500ms max)
                # The background listener is already publishing partials
                for wait_num in range(5):  # 0.5s max wait
                    await asyncio.sleep(0.1)
                    if self._current_result["text"]:
                        break

                # Return whatever we have
                final_text = self._current_result["text"]

                if final_text:
                    logger.info(f"✅ Final result: '{final_text[:100]}...'")
                    # Only publish final result if text is different from last published
                    # (background listener already published partials)
                    if on_partial_result and final_text != self._last_published_text:
                        try:
                            await on_partial_result(final_text, is_final=True)
                            self._last_published_text = final_text
                        except Exception as e:
                            logger.error(f"Error in final callback: {e}")

                    return {"text": final_text, "is_final": True, "engine": "hamza"}
                else:
                    logger.warning("⚠️ No result after 0.5s wait")
                    return {"text": "", "engine": "hamza"}

            except websockets.exceptions.ConnectionClosed:
                logger.error("❌ Connection closed")
                self.is_connected = False
                if self._listener_task:
                    self._listener_task.cancel()
                return {"text": "", "error": "Connection closed", "engine": "hamza"}

            except Exception as e:
                logger.error(f"❌ Error: {e}")
                return {"text": "", "error": str(e), "engine": "hamza"}

    async def close(self):
        """Close connection"""
        self.is_connected = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass


# Singleton
_client: Optional[HamzaClient] = None


async def get_hamza_client() -> Optional[HamzaClient]:
    """Get or create client"""
    global _client
    if _client is None:
        _client = HamzaClient()
    if not _client.is_connected:
        await _client.connect()
    return _client


async def transcribe_audio_hamza(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    is_final: bool = True,
    on_partial_result=None
) -> Dict[str, Any]:
    """
    Near-realtime transcription via Hamza STT.

    Drop-in replacement with colleague's continuous streaming architecture.

    Args:
        audio_data: Audio samples to transcribe (16kHz)
        sample_rate: Sample rate (16000 only, for API compatibility)
        is_final: Unused, kept for API compatibility
        on_partial_result: Optional async callback(text: str, is_final: bool)

    Returns:
        Dict with 'text', 'is_final', and 'engine' keys
    """
    client = await get_hamza_client()
    if client is None:
        return {"text": "", "error": "Client unavailable", "engine": "hamza"}
    return await client.transcribe(audio_data, on_partial_result=on_partial_result)


async def close_hamza_client():
    """Close client"""
    global _client
    if _client:
        await _client.close()
        _client = None
