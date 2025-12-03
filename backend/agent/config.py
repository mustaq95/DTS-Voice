import os
from dotenv import load_dotenv

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Transcription settings
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"  # Faster, good accuracy
VAD_THRESHOLD = 0.5

# Audio preprocessing settings (lightweight)
ENABLE_HIGHPASS_FILTER = os.getenv("ENABLE_HIGHPASS_FILTER", "true").lower() == "true"
ENABLE_NORMALIZATION = os.getenv("ENABLE_NORMALIZATION", "true").lower() == "true"
HIGHPASS_CUTOFF_HZ = int(os.getenv("HIGHPASS_CUTOFF_HZ", "80"))  # Hz
NORMALIZATION_TARGET_DB = float(os.getenv("NORMALIZATION_TARGET_DB", "-3.0"))  # dB

# Audio quality validation (prevents Whisper hallucinations)
ENABLE_AUDIO_VALIDATION = os.getenv("ENABLE_AUDIO_VALIDATION", "true").lower() == "true"
AUDIO_VALIDATION_RMS_THRESHOLD_DB = float(os.getenv("AUDIO_VALIDATION_RMS_THRESHOLD_DB", "-55.0"))

# Storage settings
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
MEETINGS_DIR = os.path.join(DATA_DIR, "meetings")
