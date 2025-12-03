"""
Lightweight audio preprocessing for real-time transcription.
Follows industry best practices: minimal processing, maximum stability.
"""

import numpy as np
from scipy.signal import butter, sosfilt
import logging

logger = logging.getLogger("audio-preprocessing")


def highpass_filter(
    audio: np.ndarray,
    cutoff: int = 80,
    sample_rate: int = 48000
) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to remove DC offset and low-frequency rumble.

    Args:
        audio: Input audio as float32 numpy array [-1.0, 1.0]
        cutoff: Cutoff frequency in Hz (default 80Hz removes rumble, AC hum)
        sample_rate: Sample rate of input audio

    Returns:
        Filtered audio as float32 numpy array

    Raises:
        ValueError: If audio is empty or contains invalid values
    """
    if len(audio) == 0:
        raise ValueError("Cannot filter empty audio")

    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        raise ValueError("Input audio contains NaN or Inf values")

    # 4th order Butterworth high-pass filter
    sos = butter(4, cutoff, btype='high', fs=sample_rate, output='sos')
    filtered = sosfilt(sos, audio).astype(np.float32)

    return filtered


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -3.0
) -> np.ndarray:
    """
    Peak normalize audio to target dB level.

    Args:
        audio: Input audio as float32 numpy array [-1.0, 1.0]
        target_db: Target peak level in dB (default -3dB prevents clipping)

    Returns:
        Normalized audio as float32 numpy array

    Raises:
        ValueError: If audio is empty or silent
    """
    if len(audio) == 0:
        raise ValueError("Cannot normalize empty audio")

    peak = np.max(np.abs(audio))

    # Check for near-silent audio (below -120dB)
    if peak < 1e-6:
        logger.warning(f"Audio is near-silent (peak={peak:.2e}), skipping normalization")
        return audio

    # Convert target dB to linear scale
    target_peak = 10 ** (target_db / 20.0)

    # Scale audio to target peak
    normalized = audio * (target_peak / peak)

    # Clip to [-1.0, 1.0] range (safety)
    normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)

    return normalized


def validate_audio_quality(
    audio: np.ndarray,
    sample_rate: int = 48000,
    silence_threshold_db: float = -50.0
) -> tuple[bool, str]:
    """
    Production-ready audio validation - prevents Whisper hallucinations.

    Args:
        audio: Input audio as float32 numpy array [-1.0, 1.0]
        sample_rate: Sample rate of input audio
        silence_threshold_db: Minimum acceptable RMS level in dB (default -50dB)

    Returns:
        (is_valid, reason): True if audio is good quality, False with reason if invalid
    """
    # Check 1: Empty audio
    if len(audio) == 0:
        return False, "Empty audio"

    # Check 2: NaN/Inf (corrupted data from preprocessing or buffer issues)
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        return False, "Contains NaN/Inf"

    # Check 3: Near-silence (prevents Whisper hallucinations on empty segments)
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -120

    if rms_db < silence_threshold_db:
        return False, f"Too quiet (RMS={rms_db:.1f}dB < {silence_threshold_db}dB)"

    # Check 4: Stuck buffer (all same value = corrupted buffer)
    if np.std(audio) < 1e-6:
        return False, "Constant signal (stuck buffer)"

    # Check 5: Clipping detection (prevents distorted audio)
    clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
    if clipping_ratio > 0.03:  # >3% of samples clipped
        return False, f"Excessive clipping ({clipping_ratio*100:.1f}%)"

    # Check 6: Minimum duration (prevents short noise bursts)
    duration_seconds = len(audio) / sample_rate
    if duration_seconds < 0.3:  # Less than 300ms - balanced threshold
        return False, f"Too short ({duration_seconds:.2f}s < 0.3s)"

    # All checks passed - log RMS for monitoring
    logger.debug(f"✅ Audio validation passed (RMS={rms_db:.1f}dB, duration={duration_seconds:.2f}s)")

    return True, "OK"


def preprocess_audio_frame(
    audio: np.ndarray,
    sample_rate: int = 48000,
    apply_highpass: bool = True,
    apply_normalization: bool = True
) -> np.ndarray:
    """
    Apply complete preprocessing pipeline to audio frame.

    Args:
        audio: Input audio as float32 numpy array [-1.0, 1.0]
        sample_rate: Sample rate of input audio
        apply_highpass: Whether to apply high-pass filter
        apply_normalization: Whether to apply peak normalization

    Returns:
        Preprocessed audio as float32 numpy array

    Raises:
        ValueError: If preprocessing fails with invalid audio
    """
    processed = audio.copy()

    # Step 1: High-pass filter (remove rumble, DC offset)
    if apply_highpass:
        processed = highpass_filter(processed, cutoff=80, sample_rate=sample_rate)
        logger.debug("Applied high-pass filter (80Hz)")

    # Step 2: Peak normalization (ensure consistent levels)
    if apply_normalization:
        processed = normalize_audio(processed, target_db=-3.0)
        logger.debug("Applied peak normalization (-3dB)")

    return processed
