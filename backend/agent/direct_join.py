"""
Direct room join script - bypasses automatic dispatch
Makes the agent explicitly join the 'voice-fest' room
"""
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from livekit import rtc
from livekit.plugins import silero
import numpy as np
import os
import sys
from scipy import signal as scipy_signal
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config
from agent.transcription import load_whisper_model, transcribe_audio
from api import storage

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct-join-agent")


def format_timestamp() -> str:
    """Format current time as HH:MM:SS"""
    return datetime.now().strftime("%H:%M:%S")


async def process_speech_segment(
    participant_identity: str,
    audio_buffer: list,
    meeting_id: str,
    room: rtc.Room
):
    """Process a complete speech segment"""
    if not audio_buffer:
        return

    logger.info(f"Processing speech segment from {participant_identity}, {len(audio_buffer)} frames")

    try:
        # Convert audio buffer to numpy array and resample
        # AudioStream yields AudioFrameEvent objects with .frame.data
        audio_data_48k = np.concatenate([
            np.frombuffer(frame.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            for frame in audio_buffer
        ])

        source_sample_rate = audio_buffer[0].frame.sample_rate if audio_buffer else 48000
        target_sample_rate = 16000

        if source_sample_rate != target_sample_rate:
            num_samples = int(len(audio_data_48k) * target_sample_rate / source_sample_rate)
            audio_data = scipy_signal.resample(audio_data_48k, num_samples)
        else:
            audio_data = audio_data_48k

        # Transcribe
        result = transcribe_audio(audio_data, sample_rate=target_sample_rate, is_final=True)
        text = result.get("text", "").strip()

        if not text:
            return

        timestamp = format_timestamp()
        logger.info(f"[{timestamp}] {participant_identity}: {text}")

        # Create transcript
        transcript = {
            "timestamp": timestamp,
            "speaker": participant_identity,
            "text": text,
            "is_final": True
        }

        # Publish to room
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
        logger.error(f"Error processing speech segment: {e}")


async def handle_audio_track(track: rtc.AudioTrack, participant_identity: str, meeting_id: str, room: rtc.Room):
    """Handle audio from a track"""
    logger.info(f"Handling audio track from {participant_identity}")

    audio_stream = rtc.AudioStream(track)
    audio_buffer = []

    async for frame in audio_stream:
        audio_buffer.append(frame)

        # Process buffer every ~1.5 seconds
        if len(audio_buffer) >= 75:
            await process_speech_segment(participant_identity, audio_buffer.copy(), meeting_id, room)
            audio_buffer = []


async def main():
    """Main function - directly join the room"""
    logger.info("Starting direct join agent")

    # Load Whisper model
    load_whisper_model(config.WHISPER_MODEL)

    # Create room
    room = rtc.Room()

    # Create meeting session
    meeting_id = f"voice-fest_{int(datetime.now().timestamp())}"

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track subscribed from {participant.identity}")
            asyncio.create_task(
                handle_audio_track(track, participant.identity, meeting_id, room)
            )

    # Generate token for agent
    from livekit import api
    token = api.AccessToken(
        api_key=config.LIVEKIT_API_KEY,
        api_secret=config.LIVEKIT_API_SECRET
    )
    token.with_identity("agent-transcriber")
    token.with_name("Transcription Agent")
    token.with_grants(api.VideoGrants(
        room_join=True,
        room="voice-fest",
        can_subscribe=True,
        can_publish_data=True,
    ))

    jwt_token = token.to_jwt()

    # Connect to room
    logger.info(f"Connecting to room: voice-fest")
    await room.connect(config.LIVEKIT_URL, jwt_token)
    logger.info(f"Connected! Listening for audio...")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
