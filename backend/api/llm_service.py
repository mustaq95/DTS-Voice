from openai import OpenAI
from typing import List, Dict, Any
import json

from prompts import NUDGES_PROMPT


def classify_transcripts(transcripts: List[str], api_key: str, topic: str | None = None, existing_nudges: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    """
    Classify transcript content using OpenAI

    Args:
        transcripts: List of transcript text strings
        api_key: OpenAI API key
        topic: Optional topic/segment name for enhanced context
        existing_nudges: List of already generated nudges (for deduplication)

    Returns:
        List of classified nudges
    """
    client = OpenAI(api_key=api_key, base_url="https://litellm.sandbox.dge.gov.ae/v1")

    # Join transcripts into a single string
    transcript_text = "\n".join([f"- {t}" for t in transcripts])

    # Add topic context if provided
    if topic:
        transcript_text = f"[Topic: {topic}]\n\n{transcript_text}"

    # Format existing nudges for LLM deduplication
    existing_nudges_str = ""
    if existing_nudges and len(existing_nudges) > 0:
        existing_nudges_str = "\n".join([
            f"- [{n['type']}] {n['title']}"
            for n in existing_nudges
        ])
    else:
        existing_nudges_str = "None yet"

    # Create the prompt
    prompt = NUDGES_PROMPT.format(
        transcripts=transcript_text,
        existing_nudges=existing_nudges_str
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a meeting analysis assistant that extracts structured insights from transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        # Parse the JSON response
        content = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        nudges = json.loads(content)
        return nudges

    except Exception as e:
        print(f"Error classifying transcripts: {e}")
        return []


def generate_summary(transcripts: List[Dict[str, Any]], api_key: str) -> str:
    """
    Generate a meeting summary using OpenAI

    Args:
        transcripts: List of transcript objects
        api_key: OpenAI API key

    Returns:
        Meeting summary text
    """
    client = OpenAI(api_key=api_key, base_url="https://litellm.sandbox.dge.gov.ae/v1")

    # Format transcripts
    transcript_text = "\n".join([
        f"[{t['timestamp']}] {t['speaker']}: {t['text']}"
        for t in transcripts
    ])

    prompt = f"""Summarize this meeting transcript:

{transcript_text}

Provide a concise summary covering:
1. Main topics discussed
2. Key decisions made
3. Action items identified
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a meeting summarization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""
