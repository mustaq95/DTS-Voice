#!/usr/bin/env python3
"""
Minimal Hamza WebSocket test - understand the API from scratch.

This script does the bare minimum to:
1. Connect to Hamza WebSocket
2. Send handshake
3. Send some audio
4. Print ALL messages received

Run: python test_hamza_minimal.py
"""
import asyncio
import json
import websockets
import numpy as np
from pathlib import Path
import sys

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent))
from agent import config

# Test configuration
WS_URL = config.HAMZA_WS_URL
API_KEY = config.HAMZA_API_KEY

print(f"🔧 Testing Hamza WebSocket")
print(f"   URL: {WS_URL}")
print(f"   API Key: {API_KEY[:10]}...")
print()


async def test_minimal():
    """Minimal test: connect, handshake, send audio, print responses"""

    print("1️⃣  Connecting to WebSocket...")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(WS_URL, ping_interval=None),
            timeout=10.0
        )
        print("   ✅ Connected")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return

    print()
    print("2️⃣  Sending handshake...")
    handshake = {
        "type": "handshake",
        "api_key": API_KEY,
        "options": {
            "silence_timeout": 3000,
            "sample_rate": 16000,
            "min_silence_duration_ms": 300,
            "min_speech_ms": 150,
            "client_logging": True,  # Enable server logs
            "vad_threshold": 0.2,
            "eos_enabled": True,
            "eos_threshold": 0.5,
            "audio_type": "PCM"
        }
    }

    try:
        await ws.send(json.dumps(handshake))
        print("   📤 Handshake sent")

        # Wait for ACK
        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
        data = json.loads(msg)
        print(f"   📥 Response: {data}")

        if data.get("type") != "handshake_ack":
            print(f"   ❌ Expected handshake_ack, got: {data.get('type')}")
            return

        print("   ✅ Handshake successful")
    except Exception as e:
        print(f"   ❌ Handshake failed: {e}")
        return

    print()
    print("3️⃣  Generating test audio (2s of 440Hz tone)...")
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Convert to PCM16
    audio_clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio_clipped * 32767).astype(np.int16)
    pcm_bytes = pcm16.tobytes()

    print(f"   Audio: {len(audio)} samples = {duration}s")
    print(f"   PCM16: {len(pcm_bytes)} bytes")

    print()
    print("4️⃣  Sending audio...")
    try:
        await ws.send(pcm_bytes)
        print(f"   ✅ Sent {len(pcm_bytes)} bytes")
    except Exception as e:
        print(f"   ❌ Send failed: {e}")
        return

    print()
    print("5️⃣  Waiting for responses (30s timeout)...")
    print("   (Will print ALL messages received)")
    print()

    message_count = 0
    start_time = asyncio.get_event_loop().time()

    try:
        while True:
            try:
                # Wait for message (5s timeout per message)
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                message_count += 1
                elapsed = asyncio.get_event_loop().time() - start_time

                try:
                    data = json.loads(msg)
                    msg_type = data.get('type', 'unknown')

                    print(f"   📥 Message #{message_count} ({elapsed:.1f}s): type={msg_type}")
                    print(f"      Data: {json.dumps(data, indent=6)}")
                    print()

                    # If we got transcription, we're done
                    if msg_type == "transcription":
                        text = data.get("data", {}).get("transcription", "")
                        print(f"   ✅ SUCCESS! Transcription: \"{text}\"")
                        break

                    # Check for errors
                    if msg_type == "error":
                        print(f"   ❌ Server error: {data.get('error')}")
                        break

                except json.JSONDecodeError:
                    print(f"   ⚠️  Non-JSON message: {msg[:100]}")

            except asyncio.TimeoutError:
                elapsed = asyncio.get_event_loop().time() - start_time
                print(f"   ⏱️  No message for 5s (total: {elapsed:.1f}s)...")

                if elapsed > 30:
                    print(f"   ❌ Timeout: No response after 30s")
                    break

    except Exception as e:
        print(f"   ❌ Receive error: {e}")

    print()
    print(f"6️⃣  Summary:")
    print(f"   Messages received: {message_count}")
    print(f"   Total time: {asyncio.get_event_loop().time() - start_time:.1f}s")

    # Close
    await ws.close()
    print(f"   ✅ Connection closed")


if __name__ == "__main__":
    print("=" * 60)
    print("HAMZA WEBSOCKET MINIMAL TEST")
    print("=" * 60)
    print()

    asyncio.run(test_minimal())

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
