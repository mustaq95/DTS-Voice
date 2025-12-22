"""
MLX-LM based segmentation prompt for real-time topic change detection.
Uses Qwen2.5-1.5B-Instruct-4bit for fast, on-device classification.
"""
from typing import List, Optional


def build_segmentation_prompt(
    current_topic: Optional[str],
    recent_context: List[str],
    new_transcript: str
) -> str:
    """
    Build a segmentation prompt for topic change detection.

    Args:
        current_topic: Current segment topic (None if first transcript)
        recent_context: List of recent transcripts (last 3-5) for context
        new_transcript: New transcript to classify

    Returns:
        Formatted prompt string
    """
    context_str = "\n".join(f"- {ctx}" for ctx in recent_context) if recent_context else "No prior context"
    topic_str = current_topic if current_topic else "No current topic (first transcript)"

    return f"""You are analyzing meeting transcripts to detect topic changes.

Current Topic: {topic_str}

Recent Context:
{context_str}

New Transcript: "{new_transcript}"

Task: Classify this new transcript as:
- CONTINUE: Transcript continues the current topic
- NEW_TOPIC: Transcript introduces a new topic (provide a concise topic name)
- NOISE: Transcript is noise/irrelevant (e.g., "um", "uh", background noise)

Respond ONLY with valid JSON in this exact format:
{{"action": "CONTINUE|NEW_TOPIC|NOISE", "topic": "topic name", "reason": "brief explanation"}}
"""


# Alternative: Simpler segmentation prompt template
SEGMENTATION_PROMPT = build_segmentation_prompt
