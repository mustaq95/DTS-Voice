"""
LLM Classifier Worker Process
Runs in a separate process to isolate CPU usage from main agent process.
Communicates via multiprocessing queues for non-blocking async operation.
"""
import logging
import json
import os
import multiprocessing as mp
from typing import Dict, Optional
from queue import Empty
import sys
import requests

# Add parent directory to path for imports (since this runs in separate process)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prompts import SEGMENTATION_PROMPT
from agent import config

logger = logging.getLogger("llm-classifier-worker")

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


def _load_model(model_path: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
    """
    Load the MLX classifier model (runs in worker process).

    Args:
        model_path: HuggingFace model ID or local path

    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # Import MLX-LM (only in worker process)
        from mlx_lm import load

        logger.info(f"📥 Loading classifier model: {model_path}")
        model, tokenizer = load(model_path)
        logger.info(f"✅ Classifier model loaded in worker process")
        return (model, tokenizer)

    except Exception as e:
        logger.error(f"Failed to load classifier model: {e}", exc_info=True)
        raise


def _classify_cloud(
    current_topic: Optional[str],
    recent_context: list,
    new_transcript: str,
    existing_segments: str = ""
) -> Dict[str, str]:
    """
    Classify using Qwen Cloud API (runs in worker process).

    Args:
        current_topic: Current segment topic
        recent_context: List of recent transcript texts
        new_transcript: New transcript to classify
        existing_segments: String list of already created segment topics (for deduplication)

    Returns:
        dict: Classification result
    """
    try:
        # Build prompt using centralized template
        prompt = SEGMENTATION_PROMPT(current_topic, recent_context, new_transcript, existing_segments)

        llm_logger.info("--- CLOUD LLM GENERATION START ---")
        llm_logger.info(f"Using Cloud API: {config.QWEN_CLOUD_MODEL}")
        llm_logger.info(f"Prompt:\n{prompt}")

        # Call Qwen Cloud API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.QWEN_CLOUD_API_KEY}"
        }

        payload = {
            "model": config.QWEN_CLOUD_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "settings": {
                "temperature": 0.3,
                "topP": 0.7,
                "topK": 0.8,
                "maxOutputTokens": 150,
                "thinkingConfig": {"includeThoughts": False}
            }
        }

        response = requests.post(config.QWEN_CLOUD_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            raise RuntimeError(f"API Error {response.status_code}: {response.text}")

        data = response.json()
        response_text = data["choices"][0]["message"]["content"]

        llm_logger.info(f"Raw Cloud API Response:\n{response_text}")
        llm_logger.info("--- CLOUD LLM GENERATION END ---")

        # Parse JSON response
        result = _parse_response(response_text)

        if result is None:
            logger.warning("⚠️  Cloud API parsing failed, using fallback")
            llm_logger.warning("CLOUD API PARSING FAILED - Using fallback")
            fallback = _fallback_response(current_topic)
            llm_logger.info(f"Fallback response: {fallback}")
            return fallback

        logger.info(f"✅ Cloud Classification: {result['action']} → {result['topic']} ({result['reason']})")
        llm_logger.info(f"Parsed successfully: {result}")
        return result

    except Exception as e:
        logger.error(f"❌ Cloud API classification failed: {e}", exc_info=True)
        llm_logger.error(f"CLOUD API EXCEPTION: {e}", exc_info=True)
        fallback = _fallback_response(current_topic)
        llm_logger.info(f"Using fallback due to exception: {fallback}")
        return fallback


def _classify(
    model,
    tokenizer,
    current_topic: Optional[str],
    recent_context: list,
    new_transcript: str,
    existing_segments: str = ""
) -> Dict[str, str]:
    """
    Classify a transcript using LLM (runs in worker process).

    Args:
        model: MLX model (None if cloud mode)
        tokenizer: MLX tokenizer (None if cloud mode)
        current_topic: Current segment topic
        recent_context: List of recent transcript texts
        new_transcript: New transcript to classify
        existing_segments: String list of already created segment topics (for deduplication)

    Returns:
        dict: Classification result
    """
    # Check classifier mode and route to appropriate implementation
    if config.CLASSIFIER_MODE == "cloud":
        logger.info(f"🌐 Using CLOUD classifier: {config.QWEN_CLOUD_MODEL}")
        return _classify_cloud(current_topic, recent_context, new_transcript, existing_segments)

    # Local MLX-LM mode
    logger.info(f"💻 Using LOCAL classifier: {config.CLASSIFIER_MODEL}")

    try:
        # Import MLX-LM functions
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Build prompt using centralized template
        prompt = SEGMENTATION_PROMPT(current_topic, recent_context, new_transcript, existing_segments)

        # Generate response
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

        llm_logger.info(f"Raw LLM Response:\n{response}")
        llm_logger.info("--- LLM GENERATION END ---")

        # Parse JSON response
        result = _parse_response(response)

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
        fallback = _fallback_response(current_topic)
        llm_logger.info(f"Using fallback due to exception: {fallback}")
        return fallback


def _parse_response(response: str) -> Optional[Dict[str, str]]:
    """
    Parse LLM JSON response.

    Args:
        response: Raw LLM response string

    Returns:
        dict: Parsed classification result, or None if parsing fails
    """
    try:
        # Try to find JSON in response
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

    Args:
        current_topic: Current segment topic

    Returns:
        dict: Safe CONTINUE response
    """
    return {
        "action": "CONTINUE",
        "topic": current_topic or "General Discussion",
        "reason": "LLM failed, defaulting to CONTINUE"
    }


def worker_process(request_queue: mp.Queue, response_queue: mp.Queue, model_path: str):
    """
    Worker process main loop - loads model and processes classification requests.

    Args:
        request_queue: Queue to receive classification requests
        response_queue: Queue to send classification results
        model_path: Path to MLX model
    """
    # Configure logging for worker process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )

    logger.info("🚀 LLM Classifier Worker Process starting...")
    logger.info(f"📊 Classifier Mode: {config.CLASSIFIER_MODE.upper()}")

    try:
        # Load model once at startup (only for local mode)
        if config.CLASSIFIER_MODE == "local":
            model, tokenizer = _load_model(model_path)
            logger.info(f"✅ LOCAL model ready: {model_path}")
        elif config.CLASSIFIER_MODE == "cloud":
            model, tokenizer = None, None
            logger.info(f"✅ CLOUD API ready: {config.QWEN_CLOUD_MODEL}")
        else:
            raise ValueError(f"Invalid CLASSIFIER_MODE: {config.CLASSIFIER_MODE}. Must be 'local' or 'cloud'")

        logger.info("✅ Worker process ready - waiting for classification requests")

        # Process requests in loop
        while True:
            try:
                # Get request from queue (blocking, but with timeout to allow clean shutdown)
                request = request_queue.get(timeout=1.0)

                # Check for shutdown signal
                if request is None:
                    logger.info("🛑 Received shutdown signal, exiting worker process")
                    break

                # Extract request data
                request_id = request.get("id")
                current_topic = request.get("current_topic")
                recent_context = request.get("recent_context", [])
                new_transcript = request.get("new_transcript")
                existing_segments = request.get("existing_segments", "")

                logger.info(f"📥 Processing request {request_id}: '{new_transcript[:50]}...'")

                # Classify transcript
                result = _classify(model, tokenizer, current_topic, recent_context, new_transcript, existing_segments)

                # Send response back
                response = {
                    "id": request_id,
                    "result": result
                }
                response_queue.put(response)
                logger.info(f"📤 Sent response {request_id}: {result['action']} → {result['topic']}")

            except Empty:
                # No request available, continue loop
                continue

            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)
                # Send error response
                response = {
                    "id": request_id if 'request_id' in locals() else "unknown",
                    "result": _fallback_response(None),
                    "error": str(e)
                }
                response_queue.put(response)

    except Exception as e:
        logger.error(f"Fatal error in worker process: {e}", exc_info=True)

    finally:
        logger.info("✅ Worker process shutting down cleanly")


class LLMClassifierClient:
    """
    Client for communicating with LLM classifier worker process.
    Provides async interface for segment manager.
    """

    def __init__(self, model_path: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
        """
        Initialize LLM classifier client.

        Args:
            model_path: Path to MLX model
        """
        self.model_path = model_path
        self.request_queue = mp.Queue(maxsize=100)  # Limit queue size
        self.response_queue = mp.Queue(maxsize=100)
        self.process = None
        self.request_counter = 0
        self.pending_requests = {}  # Map request_id -> callback

        logger.info(f"🔧 Initializing LLM Classifier Client with model: {model_path}")

    def start(self):
        """Start the worker process"""
        if self.process is not None:
            logger.warning("Worker process already started")
            return

        logger.info("🚀 Starting LLM classifier worker process...")
        self.process = mp.Process(
            target=worker_process,
            args=(self.request_queue, self.response_queue, self.model_path),
            daemon=True  # Daemon process will be killed when main process exits
        )
        self.process.start()
        logger.info(f"✅ Worker process started (PID: {self.process.pid})")

    def stop(self):
        """Stop the worker process"""
        if self.process is None:
            return

        logger.info("🛑 Stopping LLM classifier worker process...")
        # Send shutdown signal
        self.request_queue.put(None)
        # Wait for process to finish (with timeout)
        self.process.join(timeout=5.0)

        if self.process.is_alive():
            logger.warning("Worker process did not stop gracefully, terminating...")
            self.process.terminate()
            self.process.join(timeout=2.0)

        logger.info("✅ Worker process stopped")
        self.process = None

    async def classify_async(
        self,
        current_topic: Optional[str],
        recent_context: list,
        new_transcript: str,
        existing_segments: str = ""
    ) -> Dict[str, str]:
        """
        Classify transcript asynchronously (non-blocking).

        Args:
            current_topic: Current segment topic
            recent_context: List of recent transcript texts
            new_transcript: New transcript to classify
            existing_segments: String list of already created segment topics (for deduplication)

        Returns:
            dict: Classification result
        """
        import asyncio

        # Generate unique request ID
        self.request_counter += 1
        request_id = f"req-{self.request_counter}"

        # Create request
        request = {
            "id": request_id,
            "current_topic": current_topic,
            "recent_context": recent_context,
            "new_transcript": new_transcript,
            "existing_segments": existing_segments
        }

        # Send request to worker
        logger.info(f"📤 Sending classification request {request_id}")
        self.request_queue.put(request)

        # Poll for response (non-blocking)
        while True:
            try:
                response = self.response_queue.get_nowait()

                if response["id"] == request_id:
                    # Found our response
                    logger.info(f"📥 Received response {request_id}")
                    return response["result"]
                else:
                    # Different response - put it back and continue
                    # This shouldn't happen with proper queue ordering, but handle it
                    logger.warning(f"Received unexpected response ID: {response['id']}, expected {request_id}")

            except Empty:
                # No response yet - yield control and try again
                await asyncio.sleep(0.01)  # Small delay to avoid busy waiting

    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
