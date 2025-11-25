from typing import Callable, Optional
import asyncio


class VADHandler:
    """
    Handler for Silero VAD events
    Simple functional wrapper around VAD processing
    """

    def __init__(self, on_speech_start: Optional[Callable] = None,
                 on_speech_end: Optional[Callable] = None):
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.is_speaking = False
        self.audio_buffer = []

    async def handle_speech_start(self, participant_identity: str):
        """Called when speech starts"""
        print(f"[VAD] Speech started: {participant_identity}")
        self.is_speaking = True
        self.audio_buffer = []

        if self.on_speech_start:
            await self.on_speech_start(participant_identity)

    async def handle_speech_end(self, participant_identity: str):
        """Called when speech ends"""
        print(f"[VAD] Speech ended: {participant_identity}")
        self.is_speaking = False

        if self.on_speech_end:
            await self.on_speech_end(participant_identity, self.audio_buffer)

        self.audio_buffer = []

    def add_audio_frame(self, frame):
        """Add audio frame to buffer during speech"""
        if self.is_speaking:
            self.audio_buffer.append(frame)

    def reset(self):
        """Reset VAD state"""
        self.is_speaking = False
        self.audio_buffer = []
