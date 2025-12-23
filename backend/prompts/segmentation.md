You are analyzing meeting transcripts to detect topic changes in real-time.

Current Topic: {current_topic}

Recent Context:
{recent_context}

New Transcript: "{new_transcript}"

Already Created Segments (DO NOT DUPLICATE):
{existing_segments}

Task: Classify this new transcript as:
- **CONTINUE**: Transcript continues the current topic (DEFAULT - use this for short phrases, filler words, confirmations)
- **NEW_TOPIC**: Transcript clearly introduces a completely new topic (provide a concise topic name)
- **NOISE**: ONLY for truly unintelligible sounds, background noise, or non-speech audio

Important Guidelines:
- Short phrases like "Yes", "OK", "Excuse me", "Just one point" should be CONTINUE, NOT noise
- Filler words like "um", "uh", "like" should be CONTINUE if part of speech
- Questions or confirmations should be CONTINUE unless changing topic
- Only use NOISE for actual background sounds, static, or unintelligible audio
- When in doubt, use CONTINUE to preserve conversation flow

**CRITICAL - Deduplication:**
- Check the "Already Created Segments" list above
- If new topic is similar to an existing segment, use CONTINUE instead of NEW_TOPIC
- Only create NEW_TOPIC if truly different from ALL existing segments
- Avoid creating duplicate or very similar segment topics

**CRITICAL - Language Preservation:**
- Generate topic in the SAME language as the transcript
- If transcript is Arabic, topic MUST be in Arabic
- If transcript is English, topic MUST be in English
- NEVER translate or generate topic in a different language than the transcript

Respond ONLY with valid JSON in this exact format:
{{"action": "CONTINUE|NEW_TOPIC|NOISE", "topic": "topic name", "reason": "brief explanation"}}
