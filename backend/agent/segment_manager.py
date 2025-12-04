"""
Segment Manager for real-time transcript segmentation.
Groups transcripts into topic-based segments using LLM classification.
"""
import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict
from livekit import rtc

from agent import config
from agent.llm_classifier_worker import LLMClassifierClient

logger = logging.getLogger("segment-manager")

# Create dedicated file logger for segmentation debugging
seg_logger = logging.getLogger("segmentation-debug")
seg_logger.setLevel(logging.INFO)

# Create file handler for segmentation logs
seg_log_file = os.path.join(config.DATA_DIR, "segmentation_debug.log")
seg_file_handler = logging.FileHandler(seg_log_file, mode='a')
seg_file_handler.setLevel(logging.INFO)

# Create formatter
seg_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
seg_file_handler.setFormatter(seg_formatter)

# Add handler to logger
seg_logger.addHandler(seg_file_handler)
seg_logger.propagate = False  # Don't send to root logger

logger.info(f"📝 Segmentation debug logs will be written to: {seg_log_file}")


@dataclass
class Transcript:
    """Single transcript entry"""
    timestamp: str
    text: str
    speaker: str = ""  # Optional speaker identification


@dataclass
class Segment:
    """Topic-based segment containing multiple transcripts"""
    id: str
    topic: str
    transcripts: List[Dict]  # List of Transcript dicts
    status: str  # "active" or "completed"
    started_at: str
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class SegmentManager:
    """
    Manages real-time transcript segmentation with async LLM classification.

    Features:
    - Non-blocking: add_transcript() returns immediately
    - Async processing: LLM classification runs in background
    - Thread-safe: Uses asyncio.Lock to protect shared state
    - Persistent storage: Saves to JSON files
    - Real-time updates: Publishes to LiveKit data channel
    """

    def __init__(self, meeting_id: str, room: rtc.Room, classifier_client: LLMClassifierClient):
        """
        Initialize segment manager.

        Args:
            meeting_id: Unique meeting identifier (typically room.name)
            room: LiveKit room for publishing updates
            classifier_client: LLM classifier client (runs in separate process)
        """
        self.meeting_id = meeting_id
        self.room = room
        self.classifier_client = classifier_client

        # State
        self.segments: List[Segment] = []
        self.current_segment: Optional[Segment] = None
        self.transcript_buffer: List[Transcript] = []  # For context

        # Concurrency control
        self._lock = asyncio.Lock()
        self._tasks: List[asyncio.Task] = []

        # Storage
        self._storage_dir = os.path.join(config.DATA_DIR, "meetings", meeting_id)
        os.makedirs(self._storage_dir, exist_ok=True)

        logger.info(f"✅ SegmentManager initialized for meeting: {meeting_id}")

    def add_transcript(self, timestamp: str, text: str, speaker: str = "") -> None:
        """
        Add a new transcript for segmentation (non-blocking).

        This method returns immediately. The transcript is processed asynchronously
        in the background, including LLM classification and segment updates.

        Args:
            timestamp: Transcript timestamp (HH:MM:SS format)
            text: Transcript text
            speaker: Optional speaker identifier
        """
        transcript = Transcript(timestamp=timestamp, text=text, speaker=speaker)

        # Spawn background task (non-blocking)
        task = asyncio.create_task(self._process_transcript(transcript))
        self._tasks.append(task)

        # Cleanup completed tasks
        self._tasks = [t for t in self._tasks if not t.done()]

    async def _process_transcript(self, transcript: Transcript) -> None:
        """
        Process a transcript asynchronously (PRIVATE).

        Steps:
        1. Call LLM classifier (slow, no lock needed)
        2. Update segment state (fast, needs lock)
        3. Save to JSON and publish updates

        Args:
            transcript: Transcript to process
        """
        try:
            logger.info(f"🔄 Processing transcript for classification: '{transcript.text[:50]}...'")
            seg_logger.info("="*80)
            seg_logger.info(f"NEW TRANSCRIPT: {transcript.text}")

            # STEP 1: Classify transcript (no lock, can run concurrently)
            current_topic = self.current_segment.topic if self.current_segment else None
            recent_context = [t.text for t in self.transcript_buffer[-config.SEGMENT_CONTEXT_SIZE:]]

            logger.info(f"📋 Current topic: {current_topic}, Context size: {len(recent_context)}")
            seg_logger.info(f"Current Topic: {current_topic}")
            seg_logger.info(f"Context ({len(recent_context)} transcripts): {recent_context}")

            seg_logger.info("Calling LLM classifier (in separate process)...")
            classification = await self.classifier_client.classify_async(
                current_topic=current_topic,
                recent_context=recent_context,
                new_transcript=transcript.text
            )

            logger.info(f"✅ Classification result: {classification['action']} → {classification['topic']} ({classification['reason']})")
            seg_logger.info(f"CLASSIFICATION RESULT:")
            seg_logger.info(f"  Action: {classification['action']}")
            seg_logger.info(f"  Topic: {classification['topic']}")
            seg_logger.info(f"  Reason: {classification['reason']}")

            # STEP 2: Update state (needs lock)
            async with self._lock:
                await self._handle_classification(transcript, classification)

            # STEP 3: Publish update (no lock needed)
            await self._publish_segment_update()

        except Exception as e:
            logger.error(f"❌ Error processing transcript: {e}", exc_info=True)
            seg_logger.error(f"ERROR: {e}", exc_info=True)

    async def _handle_classification(
        self,
        transcript: Transcript,
        classification: Dict[str, str]
    ) -> None:
        """
        Handle classification result and update segments (PRIVATE, requires lock).

        Args:
            transcript: Original transcript
            classification: LLM classification result
        """
        action = classification.get("action", "CONTINUE")
        topic = classification.get("topic", "General Discussion")
        reason = classification.get("reason", "")

        if action == "NOISE":
            # Filter out noise
            logger.debug(f"🔇 Filtered noise: '{transcript.text}' ({reason})")
            await self._publish_noise_filtered(transcript, reason)
            return

        elif action == "NEW_TOPIC":
            # Check if current segment is long enough for topic change
            if self.current_segment is not None:
                word_count = sum(len(t["text"].split()) for t in self.current_segment.transcripts)

                if word_count >= config.SEGMENT_MIN_WORDS:
                    # Segment is long enough - complete it and start new one
                    self._complete_current_segment()
                    await self._publish_segment_complete(self.segments[-1])
                    self._start_new_segment(topic, transcript)
                    logger.info(f"📌 New segment: '{topic}' ({reason})")
                else:
                    # Segment too short - merge into current and update topic
                    self.current_segment.topic = topic
                    self.current_segment.transcripts.append(asdict(transcript))
                    logger.info(f"🔄 Updated segment topic to '{topic}' (too short for split: {word_count} < {config.SEGMENT_MIN_WORDS} words)")
            else:
                # No current segment - start new one
                self._start_new_segment(topic, transcript)
                logger.info(f"📌 New segment: '{topic}' ({reason})")

        else:  # CONTINUE
            # Add to current segment
            if self.current_segment is None:
                # First transcript - create initial segment
                self._start_new_segment(topic, transcript)
                logger.info(f"📌 Initial segment: '{topic}'")
            else:
                # Append to current segment
                self.current_segment.transcripts.append(asdict(transcript))
                logger.debug(f"➕ Added to segment '{self.current_segment.topic}'")

        # Update transcript buffer (keep last N for context)
        self.transcript_buffer.append(transcript)
        if len(self.transcript_buffer) > config.SEGMENT_CONTEXT_SIZE * 2:
            self.transcript_buffer = self.transcript_buffer[-config.SEGMENT_CONTEXT_SIZE * 2:]

        # Check if segment is too long (force new segment)
        if self.current_segment:
            word_count = sum(len(t["text"].split()) for t in self.current_segment.transcripts)
            if word_count > config.SEGMENT_MAX_WORDS:
                logger.info(f"✂️  Segment too long ({word_count} words), forcing new segment")

                # Keep topic before completing
                continued_topic = self.current_segment.topic

                self._complete_current_segment()
                await self._publish_segment_complete(self.segments[-1])

                # Start new segment with same topic
                self._start_new_segment(continued_topic, transcript)

        # Save to JSON
        self._save_to_json()

    def _start_new_segment(self, topic: str, initial_transcript: Transcript) -> None:
        """Start a new segment (PRIVATE, requires lock)"""
        self.current_segment = Segment(
            id=f"seg-{uuid.uuid4().hex[:8]}",
            topic=topic,
            transcripts=[asdict(initial_transcript)],
            status="active",
            started_at=datetime.now().isoformat()
        )

    def _complete_current_segment(self) -> None:
        """Complete the current segment and add to segments list (PRIVATE, requires lock)"""
        if self.current_segment is None:
            return

        self.current_segment.status = "completed"
        self.current_segment.completed_at = datetime.now().isoformat()
        self.segments.append(self.current_segment)
        self.current_segment = None

    def _save_to_json(self) -> None:
        """Save segments to JSON file (PRIVATE, requires lock)"""
        try:
            filepath = os.path.join(self._storage_dir, "segments.json")

            state = {
                "meeting_id": self.meeting_id,
                "segments": [seg.to_dict() for seg in self.segments],
                "current_segment": self.current_segment.to_dict() if self.current_segment else None
            }

            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"💾 Saved segments to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save segments: {e}", exc_info=True)

    async def _publish_segment_update(self) -> None:
        """Publish buffer_update to LiveKit data channel"""
        try:
            if self.current_segment is None:
                return

            message = {
                "type": "buffer_update",
                "data": self.current_segment.to_dict()
            }

            await self.room.local_participant.publish_data(
                payload=json.dumps(message).encode('utf-8'),
                reliable=True,
                topic="segments"
            )

            logger.debug(f"📡 Published buffer_update")

        except Exception as e:
            logger.error(f"Failed to publish segment update: {e}", exc_info=True)

    async def _publish_segment_complete(self, segment: Segment) -> None:
        """Publish segment_complete to LiveKit data channel"""
        try:
            message = {
                "type": "segment_complete",
                "data": segment.to_dict()
            }

            await self.room.local_participant.publish_data(
                payload=json.dumps(message).encode('utf-8'),
                reliable=True,
                topic="segments"
            )

            logger.info(f"📡 Published segment_complete: '{segment.topic}'")

        except Exception as e:
            logger.error(f"Failed to publish segment complete: {e}", exc_info=True)

    async def _publish_noise_filtered(self, transcript: Transcript, reason: str) -> None:
        """Publish noise_filtered to LiveKit data channel"""
        try:
            message = {
                "type": "noise_filtered",
                "data": {
                    "transcript": asdict(transcript),
                    "reason": reason
                }
            }

            await self.room.local_participant.publish_data(
                payload=json.dumps(message).encode('utf-8'),
                reliable=True,
                topic="segments"
            )

            logger.debug(f"📡 Published noise_filtered")

        except Exception as e:
            logger.error(f"Failed to publish noise filtered: {e}", exc_info=True)

    def get_state(self) -> Dict:
        """
        Get current segmentation state (thread-safe).

        Returns:
            dict: Current state with segments and active buffer
        """
        # No lock needed for read-only access (Python GIL protects simple reads)
        return {
            "meeting_id": self.meeting_id,
            "segments": [seg.to_dict() for seg in self.segments],
            "current_segment": self.current_segment.to_dict() if self.current_segment else None
        }

    async def cleanup(self) -> None:
        """
        Cleanup: Cancel pending tasks and complete current segment.
        Call this when the meeting ends or agent disconnects.
        """
        logger.info(f"🧹 Cleaning up SegmentManager for meeting: {self.meeting_id}")

        # Cancel pending tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete/cancel
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Complete current segment
        async with self._lock:
            if self.current_segment is not None:
                self._complete_current_segment()
                self._save_to_json()

                if self.segments:
                    await self._publish_segment_complete(self.segments[-1])

        logger.info(f"✅ SegmentManager cleanup complete")
