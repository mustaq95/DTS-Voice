"""
OpenAI-based classification prompt for extracting structured insights from transcripts.
Used for identifying key proposals, delivery risks, and action items.
"""

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
