"""
Video Translator - A complete video translation pipeline using Qwen3 models.

This package provides tools for:
- Speech-to-text transcription (Qwen3-ASR)
- Text-to-speech synthesis (Qwen3-TTS)
- Forced alignment for timestamps (Qwen3-ForcedAligner)
- Video audio extraction and muxing (FFmpeg)
- Full video translation pipeline
"""

__version__ = "0.1.0"
__author__ = "Video Translator Team"

from .config import Config, get_config
from .pipeline import VideoTranslator, TranslationResult, TranscriptionResult

__all__ = [
    "VideoTranslator",
    "Config",
    "get_config",
    "TranslationResult",
    "TranscriptionResult",
    "__version__",
]
