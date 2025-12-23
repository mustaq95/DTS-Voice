You are analyzing a meeting transcript to extract structured insights.

Recent transcript excerpts:
{transcripts}

Already Generated Nudges (DO NOT DUPLICATE):
{existing_nudges}

Please classify the content into the following categories and provide structured output:

1. **Key Proposal**: Important proposals, decisions, or facts mentioned
2. **Delivery Risk**: Concerns, risks, or blockers discussed
3. **Action Item**: Tasks or actions that need to be completed

For each category found, provide:
- title: A concise summary (max 15 words)
- quote: The relevant excerpt from the transcript
- confidence: Your confidence score (0.0 to 1.0)

**CRITICAL - Deduplication:**
- Check the "Already Generated Nudges" list above
- DO NOT create duplicate or similar nudges
- If a nudge is already generated, return empty array

Return ONLY a JSON array with this structure:
[
  {{
    "type": "key_proposal" | "delivery_risk" | "action_item",
    "title": "...",
    "quote": "...",
    "confidence": 0.0-1.0
  }}
]

If no significant items are found OR would be duplicate, return empty array: []
Return only the JSON array, no other text.
