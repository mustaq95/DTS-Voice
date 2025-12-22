"""
Centralized prompt management for LLM-based features.
"""

from .classification import CLASSIFICATION_PROMPT
from .segmentation import SEGMENTATION_PROMPT

__all__ = [
    "CLASSIFICATION_PROMPT",
    "SEGMENTATION_PROMPT",
]
