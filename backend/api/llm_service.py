from openai import OpenAI
from typing import List, Dict, Any
import json


CLASSIFICATION_PROMPT = """You are analyzing a meeting transcript to extract structured insights.

Recent transcript excerpts:
{transcripts}

Please classify the content into the following categories and provide structured output:

1. **Key Proposal**: Important proposals, decisions, or facts mentioned
2. **Delivery Risk**: Concerns, risks, or blockers discussed
3. **Action Item**: Tasks or actions that need to be completed

For each category found, provide:
- title: A concise summary (max 15 words)
- quote: The relevant excerpt from the transcript
- confidence: Your confidence score (0.0 to 1.0)

Return ONLY a JSON array with this structure:
[
  {{
    "type": "key_proposal" | "delivery_risk" | "action_item",
    "title": "...",
    "quote": "...",
    "confidence": 0.0-1.0
  }}
]

If no significant items are found in a category, omit it from the output.
Return only the JSON array, no other text.
"""


def classify_transcripts(transcripts: List[str], api_key: str) -> List[Dict[str, Any]]:
    """
    Classify transcript content using OpenAI

    Args:
        transcripts: List of transcript text strings
        api_key: OpenAI API key

    Returns:
        List of classified nudges
    """
    client = OpenAI(api_key=api_key, base_url="https://litellm.sandbox.dge.gov.ae/v1")

    # Join transcripts into a single string
    transcript_text = "\n".join([f"- {t}" for t in transcripts])

    # Create the prompt
    prompt = CLASSIFICATION_PROMPT.format(transcripts=transcript_text)

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
