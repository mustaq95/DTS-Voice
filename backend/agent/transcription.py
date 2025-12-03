import mlx_whisper
from typing import Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger("meeting-agent")

# Global model instance
_whisper_model = None


def load_whisper_model(model_name: str = "mlx-community/whisper-tiny"):
    """Load MLX Whisper model"""
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading MLX Whisper {model_name} model...")
        _whisper_model = model_name
    return _whisper_model


def transcribe_audio(audio_data: np.ndarray, sample_rate: int = 16000, is_final: bool = True) -> Dict[str, Any]:
    """
    Transcribe audio using MLX Whisper

    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate of audio (default 16000)
        is_final: Whether this is a final transcription or interim

    Returns:
        Dictionary with transcription results
    """
    try:
        # Ensure model is loaded
        model = load_whisper_model()

        # Run transcription with anti-hallucination parameters
        result = mlx_whisper.transcribe(
            audio_data,
            path_or_hf_repo=model,
            language=None,                      # Auto-detect language (important!)
            task="transcribe",                  # Keep original language (not translate)
            fp16=True,                          # Use FP16 for speed on Apple Silicon
            condition_on_previous_text=False,   # Prevents hallucination loops
        )

        text = result.get("text", "").strip()

        return {
            "text": text,
            "is_final": is_final,
            "language": result.get("language", "en"),
            "segments": result.get("segments", [])
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return {
            "text": "",
            "is_final": is_final,
            "error": str(e)
        }


def audio_buffer_to_numpy(audio_frames, sample_rate: int = 16000) -> np.ndarray:
    """
    Convert audio frames to numpy array for Whisper

    Args:
        audio_frames: Audio frames from LiveKit
        sample_rate: Sample rate

    Returns:
        Numpy array of audio samples
    """
    # Combine frames into a single buffer
    # Note: Actual implementation depends on LiveKit frame format
    # This is a placeholder that needs to be adapted to LiveKit's actual audio format

    # For now, return a simple conversion
    # In production, you'd need to properly handle LiveKit's AudioFrame format
    if isinstance(audio_frames, list):
        # Concatenate multiple frames
        audio_data = np.concatenate([np.frombuffer(frame, dtype=np.float32) for frame in audio_frames])
    else:
        audio_data = np.frombuffer(audio_frames, dtype=np.float32)

    return audio_data
