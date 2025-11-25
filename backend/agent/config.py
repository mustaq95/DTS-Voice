import os
from dotenv import load_dotenv

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Transcription settings
WHISPER_MODEL = "mlx-community/whisper-small-mlx"  # HuggingFace model path
VAD_THRESHOLD = 0.5

# Storage settings
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
MEETINGS_DIR = os.path.join(DATA_DIR, "meetings")
