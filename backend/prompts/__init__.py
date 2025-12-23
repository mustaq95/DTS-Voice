"""
Centralized prompt management for LLM-based features.
Prompts are stored as markdown files for easy editing.
"""
import os
from typing import List, Optional

# Get prompts directory
PROMPTS_DIR = os.path.dirname(__file__)


def _read_prompt(filename: str) -> str:
    """Read a prompt from a markdown file."""
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# Load nudges prompt (for OpenAI-based nudge extraction)
NUDGES_PROMPT = _read_prompt('nudges.md')

def SEGMENTATION_PROMPT(
    current_topic: Optional[str],
    recent_context: List[str],
    new_transcript: str,
    existing_segments: str = ""
) -> str:
    """
    Build a segmentation prompt for topic change detection.

    Args:
        current_topic: Current segment topic (None if first transcript)
        recent_context: List of recent transcripts (last 3-5) for context
        new_transcript: New transcript to classify
        existing_segments: String list of already created segment topics (for deduplication)

    Returns:
        Formatted prompt string
    """
    # Format context and topic strings
    context_str = "\n".join(f"- {ctx}" for ctx in recent_context) if recent_context else "No prior context"
    topic_str = current_topic if current_topic else "No current topic (first transcript)"
    segments_str = existing_segments if existing_segments else "None yet"

    # Load template and format it
    template = _read_prompt('segmentation.md')
    return template.format(
        current_topic=topic_str,
        recent_context=context_str,
        new_transcript=new_transcript,
        existing_segments=segments_str
    )


__all__ = [
    "NUDGES_PROMPT",
    "SEGMENTATION_PROMPT",
]
