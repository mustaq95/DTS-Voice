"""
Simple Hamza WebSocket STT Client
Streams audio and publishes results immediately as they arrive
"""
import asyncio
import json
import logging
import numpy as np
import websockets
from typing import Optional, Dict, Any

try:
    from . import config
except ImportError:
    class config:
        HAMZA_WS_URL = "wss://beta.govgpt.gov.ae/llm/hamsa/ws"
        HAMZA_API_KEY = ""
        HAMZA_SILENCE_TIMEOUT_MS = 3000
        HAMZA_MIN_SILENCE_MS = 300
        HAMZA_MIN_SPEECH_MS = 150
        HAMZA_VAD_THRESHOLD = 0.2
        HAMZA_EOS_ENABLED = True
        HAMZA_EOS_THRESHOLD = 0.6

logger = logging.getLogger("hamza-stt")


class HamzaClient:
    """Simple streaming client - sends audio, returns result immediately"""

    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self._lock = asyncio.Lock()

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
                    "audio_type": "PCM",
                }
            }
            await self.ws.send(json.dumps(handshake))

            msg = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(msg)

            if data.get("type") == "handshake_ack":
                logger.info("✅ Hamza connected")
                self.is_connected = True
                return True
            
            logger.error(f"❌ Handshake failed: {data}")
            return False

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    async def transcribe(self, audio_data: np.ndarray, on_partial_result=None) -> Dict[str, Any]:
        """
        Send audio and get transcription result with real-time partial updates.

        Args:
            audio_data: Audio samples to transcribe (16kHz)
            on_partial_result: Optional async callback(text: str, is_final: bool) for real-time updates

        Returns:
            Dict with 'text', 'is_final', and 'engine' keys
        """
        async with self._lock:
            try:
                # Ensure connected
                if not self.is_connected or not self.ws:
                    if not await self.connect():
                        return {"text": "", "error": "Connection failed", "engine": "hamza"}

                # Add 500ms silence padding at both ends for better VAD/EOS detection
                silence_samples = np.zeros(int(16000 * 0.5), dtype=np.float32)
                audio_with_silence = np.concatenate([silence_samples, audio_data, silence_samples])

                # Convert to PCM16
                audio_clipped = np.clip(audio_with_silence, -1.0, 1.0)
                pcm_bytes = (audio_clipped * 32767).astype(np.int16).tobytes()

                duration_s = len(audio_data) / 16000
                logger.info(f"🎙️ Sending {duration_s:.1f}s audio (+1.0s silence padding)")

                # Stream audio in 256-byte chunks
                chunk_size = 256
                result_text = ""
                total_chunks = len(pcm_bytes) // chunk_size + (1 if len(pcm_bytes) % chunk_size else 0)
                logger.info(f"📦 Sending {total_chunks} chunks ({chunk_size} bytes each)")

                for i in range(0, len(pcm_bytes), chunk_size):
                    chunk_num = i // chunk_size + 1
                    chunk = pcm_bytes[i:i + chunk_size]
                    await self.ws.send(chunk)

                    # Check for results during streaming
                    try:
                        msg = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                        data = json.loads(msg)

                        if data.get("type") == "transcription":
                            text = data.get("data", {}).get("transcription", "").strip()
                            if text:
                                result_text = text
                                logger.info(f"📝 Got partial at chunk {chunk_num}/{total_chunks}")
                                logger.info(f"🔍 Text: '{text}...' (length: {len(text)} chars)")

                                # Publish partial result to UI in real-time
                                is_final = (chunk_num == total_chunks)
                                if on_partial_result:
                                    try:
                                        await on_partial_result(text, is_final=is_final)
                                    except Exception as e:
                                        logger.error(f"Error in partial result callback: {e}")

                                # If last chunk, return immediately (even if callback failed)
                                if is_final:
                                    logger.info(f"✅ Last chunk processed, returning final result")
                                    return {"text": text, "is_final": True, "engine": "hamza"}

                    except asyncio.TimeoutError:
                        pass  # Continue streaming

                # Wait for final result after all chunks sent (trailing silence triggers EOS)
                logger.info("⏳ All chunks sent, waiting for final result...")
                for wait_num in range(15):  # 1.5s max wait
                    try:
                        msg = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                        data = json.loads(msg)
                        if data.get("type") == "transcription":
                            text = data.get("data", {}).get("transcription", "").strip()
                            if text:
                                logger.info(f"✅ Got final result after {(wait_num+1)*100}ms wait")
                                logger.info(f"🔍 Text: '{text[:100]}...' (length: {len(text)} chars)")

                                # Publish final result via callback
                                if on_partial_result:
                                    try:
                                        await on_partial_result(text, is_final=True)
                                    except Exception as e:
                                        logger.error(f"Error in final result callback: {e}")

                                return {"text": text, "is_final": True, "engine": "hamza"}
                    except asyncio.TimeoutError:
                        continue

                # If no final result arrived, return the last partial result we got
                # Note: Don't call callback here - this text was already published as partial
                if result_text:
                    logger.warning("⚠️ No new result after waiting, returning last partial result")
                    return {"text": result_text, "is_final": True, "engine": "hamza"}

                logger.warning("⚠️ No result after sending audio+silence (waited 1.5s)")
                return {"text": "", "engine": "hamza"}

            except websockets.exceptions.ConnectionClosed:
                logger.error("❌ Connection closed")
                self.is_connected = False
                return {"text": "", "error": "Connection closed", "engine": "hamza"}

            except Exception as e:
                logger.error(f"❌ Error: {e}")
                return {"text": "", "error": str(e), "engine": "hamza"}

    async def close(self):
        """Close connection"""
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
        self.is_connected = False


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
    Transcribe audio with real-time partial results via Hamza STT.

    Args:
        audio_data: Audio samples to transcribe (16kHz)
        sample_rate: Sample rate (16000 only, for API compatibility)
        is_final: Unused, kept for API compatibility
        on_partial_result: Optional async callback(text: str, is_final: bool) for streaming updates

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