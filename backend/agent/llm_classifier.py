"""
LLM-based transcript classifier for topic segmentation.
Uses MLX-optimized Qwen2.5-1.5B-Instruct-4bit for real-time classification.
"""
import logging
import json
import os
from typing import List, Dict, Optional

from prompts import SEGMENTATION_PROMPT

# LAZY IMPORT: Move heavy MLX-LM imports inside functions to avoid slow module load
# from mlx_lm import load, stream_generate
# from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger("llm-classifier")

# Get data directory from environment or use default
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# Create dedicated file logger for LLM responses
llm_logger = logging.getLogger("llm-debug")
llm_logger.setLevel(logging.INFO)

# Create file handler
llm_log_file = os.path.join(DATA_DIR, "llm_classifier_debug.log")
llm_file_handler = logging.FileHandler(llm_log_file, mode='a')
llm_file_handler.setLevel(logging.INFO)

# Create formatter
llm_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
llm_file_handler.setFormatter(llm_formatter)

# Add handler
llm_logger.addHandler(llm_file_handler)
llm_logger.propagate = False

logger.info(f"📝 LLM classifier debug logs will be written to: {llm_log_file}")

# Global model cache
_model = None
_tokenizer = None


def load_classifier_model(model_path: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
    """
    Load and cache the MLX classifier model.

    Args:
        model_path: HuggingFace model ID or local path

    Returns:
        tuple: (model, tokenizer) cached for reuse
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        logger.info(f"✅ Using cached classifier model: {model_path}")
        return (_model, _tokenizer)

    try:
        # Lazy import: Only import MLX-LM when actually loading the model
        from mlx_lm import load

        logger.info(f"📥 Loading classifier model: {model_path}")
        _model, _tokenizer = load(model_path)
        logger.info(f"✅ Classifier model loaded successfully")
        return (_model, _tokenizer)

    except Exception as e:
        logger.error(f"Failed to load classifier model: {e}", exc_info=True)
        raise


async def classify_transcript(
    current_topic: Optional[str],
    recent_context: List[str],
    new_transcript: str,
    model_tuple: Optional[tuple] = None
) -> Dict[str, str]:
    """
    Classify a transcript as CONTINUE/NEW_TOPIC/NOISE using LLM.

    Args:
        current_topic: Current segment topic (None if first transcript)
        recent_context: List of recent transcripts (last 3-5) for context
        new_transcript: New transcript to classify
        model_tuple: Optional (model, tokenizer) tuple. If None, uses cached model.

    Returns:
        dict: {
            "action": "CONTINUE" | "NEW_TOPIC" | "NOISE",
            "topic": str,  # Suggested topic name (for NEW_TOPIC)
            "reason": str  # Brief explanation
        }
    """
    logger.info(f"🤖 classify_transcript() called - Topic: '{current_topic}', Transcript: '{new_transcript[:50]}...'")

    # Use provided model or load from cache
    if model_tuple is None:
        logger.info("📥 Loading classifier model from cache")
        model, tokenizer = load_classifier_model()
    else:
        logger.info("✅ Using provided classifier model")
        model, tokenizer = model_tuple

    # Build prompt using centralized template
    prompt = SEGMENTATION_PROMPT(current_topic, recent_context, new_transcript)

    try:
        # Lazy import: Only import MLX-LM functions when actually generating
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Generate response
        logger.info("🔄 Generating LLM response...")
        llm_logger.info("--- LLM GENERATION START ---")
        llm_logger.info(f"Prompt:\n{prompt}")

        messages = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Create sampler with low temperature for deterministic classification
        sampler = make_sampler(temp=0.3)

        # Use stream_generate - accumulate tokens to build full response
        response = ""
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt_formatted,
            max_tokens=150,
            sampler=sampler
        ):
            # Each chunk contains a single token - accumulate them
            if hasattr(chunk, 'text'):
                response += chunk.text

        logger.info(f"📝 LLM raw response: {response[:200]}...")
        llm_logger.info(f"Raw LLM Response:\n{response}")
        llm_logger.info("--- LLM GENERATION END ---")

        # Parse JSON response
        result = _parse_llm_response(response)

        if result is None:
            # Parsing failed - use fallback
            logger.warning("⚠️  Parsing failed, using fallback response")
            llm_logger.warning("PARSING FAILED - Using fallback")
            fallback = _fallback_response(current_topic)
            llm_logger.info(f"Fallback response: {fallback}")
            return fallback

        logger.info(f"✅ Classification: {result['action']} → {result['topic']} ({result['reason']})")
        llm_logger.info(f"Parsed successfully: {result}")
        return result

    except Exception as e:
        logger.error(f"❌ LLM classification failed: {e}", exc_info=True)
        llm_logger.error(f"EXCEPTION: {e}", exc_info=True)
        # Fallback to safe default
        fallback = _fallback_response(current_topic)
        llm_logger.info(f"Using fallback due to exception: {fallback}")
        return fallback


def _parse_llm_response(response: str) -> Optional[Dict[str, str]]:
    """
    Parse LLM JSON response.

    Args:
        response: Raw LLM response string

    Returns:
        dict: Parsed classification result, or None if parsing fails
    """
    try:
        # Try to find JSON in response (model might add extra text)
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")

        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)

        # Validate required fields
        if "action" not in result or "topic" not in result or "reason" not in result:
            raise ValueError("Missing required fields in JSON")

        # Validate action value
        if result["action"] not in ["CONTINUE", "NEW_TOPIC", "NOISE"]:
            raise ValueError(f"Invalid action: {result['action']}")

        return result

    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}. Response: {response[:200]}")
        return None


def _fallback_response(current_topic: Optional[str]) -> Dict[str, str]:
    """
    Simple fallback response when LLM fails.
    Always returns CONTINUE with current topic to avoid losing transcripts.

    Args:
        current_topic: Current segment topic (None if first transcript)

    Returns:
        dict: Safe CONTINUE response
    """
    return {
        "action": "CONTINUE",
        "topic": current_topic or "General Discussion",
        "reason": "LLM failed, defaulting to CONTINUE"
    }
