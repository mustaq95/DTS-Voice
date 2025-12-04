import os
from dotenv import load_dotenv

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Transcription settings
# M4 OPTIMIZED: Turbo model with M4 Neural Engine optimizations
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"  # Turbo model - fast & accurate on M4
VAD_THRESHOLD = 0.35  # M4 OPTIMIZED: Lower threshold for continuous speech

# Audio preprocessing settings (lightweight)
# BALANCED: High-pass enabled, normalization disabled
ENABLE_HIGHPASS_FILTER = os.getenv("ENABLE_HIGHPASS_FILTER", "true").lower() == "true"
ENABLE_NORMALIZATION = os.getenv("ENABLE_NORMALIZATION", "false").lower() == "true"
HIGHPASS_CUTOFF_HZ = int(os.getenv("HIGHPASS_CUTOFF_HZ", "100"))  # Hz - removes rumble/hum
NORMALIZATION_TARGET_DB = float(os.getenv("NORMALIZATION_TARGET_DB", "-3.0"))  # dB - prevents clipping

# Audio quality validation (prevents Whisper hallucinations)
# M4 OPTIMIZED: More permissive threshold to capture quieter speech segments
ENABLE_AUDIO_VALIDATION = os.getenv("ENABLE_AUDIO_VALIDATION", "true").lower() == "true"
AUDIO_VALIDATION_RMS_THRESHOLD_DB = float(os.getenv("AUDIO_VALIDATION_RMS_THRESHOLD_DB", "-55.0"))  # M4 OPTIMIZED

# Segmentation settings
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
SEGMENT_MIN_WORDS = int(os.getenv("SEGMENT_MIN_WORDS", "20"))  # Min words before topic change allowed
SEGMENT_MAX_WORDS = int(os.getenv("SEGMENT_MAX_WORDS", "500"))  # Force new segment if exceeded
SEGMENT_CONTEXT_SIZE = int(os.getenv("SEGMENT_CONTEXT_SIZE", "5"))  # Recent transcripts for LLM context

# Storage settings
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
MEETINGS_DIR = os.path.join(DATA_DIR, "meetings")
