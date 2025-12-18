"""
Shared model configuration for API-Agent communication.

Since LiveKit data messages don't reach agents when sent via server API,
we use a file-based approach for model switching.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("model-config")

# Path to model config file
CONFIG_DIR = Path(__file__).parent.parent / "data"
CONFIG_FILE = CONFIG_DIR / "model_config.json"


def ensure_config_dir():
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def save_model_config(room_name: str, model: str) -> None:
    """
    Save model configuration for a room.

    Args:
        room_name: Room identifier
        model: Model name (e.g., 'mlx_whisper', 'hamza')
    """
    ensure_config_dir()

    # Load existing config
    config = load_all_model_configs()

    # Update room config
    config[room_name] = model

    # Save to file
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.debug(f"✅ Saved model config: {room_name} → {model}")
    except Exception as e:
        logger.error(f"❌ Failed to save model config: {e}")


def load_model_config(room_name: str, default: str = "mlx_whisper") -> str:
    """
    Load model configuration for a room.

    Args:
        room_name: Room identifier
        default: Default model if not found

    Returns:
        str: Model name
    """
    config = load_all_model_configs()
    model = config.get(room_name, default)
    logger.debug(f"📖 Loaded model config for {room_name}: {model}")
    return model


def load_all_model_configs() -> Dict[str, str]:
    """
    Load all model configurations.

    Returns:
        dict: Room name → model mapping
    """
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️  Failed to load model config: {e}")
        return {}


def delete_model_config(room_name: str) -> None:
    """
    Delete model configuration for a room (cleanup).

    Args:
        room_name: Room identifier
    """
    config = load_all_model_configs()

    if room_name in config:
        del config[room_name]

        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            logger.debug(f"🗑️  Deleted model config for {room_name}")
        except Exception as e:
            logger.error(f"❌ Failed to delete model config: {e}")
