"""Configuration management for Video Translator."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration for Video Translator application.
    
    Can be configured via environment variables or .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ==================== Model Configuration ====================
    
    qwen_asr_model: str = Field(
        default="Qwen/Qwen3-ASR-1.7B",
        description="HuggingFace model ID for ASR",
    )
    
    qwen_tts_model: str = Field(
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        description="HuggingFace model ID for TTS (CustomVoice mode)",
    )
    
    qwen_tts_base_model: str = Field(
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        description="HuggingFace model ID for TTS Base model (for voice cloning)",
    )
    
    qwen_aligner_model: str = Field(
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        description="HuggingFace model ID for Forced Aligner",
    )
    
    # ==================== Hardware Configuration ====================
    
    device: Literal["cuda", "mps", "cpu", "auto"] = Field(
        default="auto",
        description="Device to run models on (cuda, mps, cpu, auto)",
    )
    
    precision: Literal["bf16", "fp16", "fp32"] = Field(
        default="bf16",
        description="Model precision (bf16, fp16, fp32)",
    )
    
    flash_attention: bool = Field(
        default=True,
        description="Enable FlashAttention 2 for faster inference",
    )
    
    # ==================== Storage Configuration ====================
    
    model_cache_dir: str = Field(
        default="./models_cache",
        description="Directory to cache downloaded models",
    )
    
    output_dir: str = Field(
        default="./output",
        description="Directory for output files",
    )
    
    temp_dir: str = Field(
        default="/tmp/video_translator",
        description="Directory for temporary files",
    )
    
    # ==================== API Configuration ====================
    
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    
    api_port: int = Field(
        default=8000,
        description="API server port",
    )
    
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (optional)",
    )
    
    # ==================== Queue Configuration ====================
    
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for task queue",
    )
    
    celery_workers: int = Field(
        default=2,
        description="Number of Celery workers",
    )
    
    # ==================== Translation Configuration ====================
    
    target_language: str = Field(
        default="es",
        description="Default target language code (ISO 639-1)",
    )
    
    # ==================== Processing Configuration ====================
    
    audio_sample_rate: int = Field(
        default=16000,
        description="Sample rate for audio processing",
    )
    
    audio_channels: int = Field(
        default=1,
        description="Number of audio channels (1=mono, 2=stereo)",
    )
    
    max_audio_duration: int = Field(
        default=1200,  # 20 minutes
        description="Maximum audio duration in seconds per segment",
    )
    
    use_vad: bool = Field(
        default=True,
        description="Enable VAD-based segmentation before ASR",
    )

    vad_threshold: float = Field(
        default=0.5,
        description="Voice Activity Detection threshold",
    )

    vad_min_speech_duration_ms: int = Field(
        default=250,
        description="Minimum speech region duration for VAD, in milliseconds",
    )

    vad_min_silence_duration_ms: int = Field(
        default=200,
        description="Minimum silence to split VAD regions, in milliseconds",
    )

    max_segment_duration: float = Field(
        default=30.0,
        description="Maximum segment duration (seconds) after VAD grouping",
    )

    min_segment_duration: float = Field(
        default=0.4,
        description="Minimum segment duration (seconds) to process",
    )

    duration_error_tolerance: float = Field(
        default=0.15,
        description="Acceptable synthesized duration error ratio",
    )

    max_retiming_ratio: float = Field(
        default=1.2,
        description="Maximum mild retiming ratio for TTS outputs",
    )

    max_translation_retries: int = Field(
        default=2,
        description="Number of translation compression retries on timing mismatch",
    )

    segment_extract_workers: int = Field(
        default=0,
        description=(
            "CPU workers for parallel FFmpeg segment extraction "
            "(0 = auto)"
        ),
    )

    keep_background_audio: bool = Field(
        default=False,
        description="Mix original background audio under synthesized translated speech",
    )

    background_audio_volume: float = Field(
        default=0.2,
        description="Background audio mix gain (0.0 to 1.0)",
    )

    embed_subtitles: bool = Field(
        default=False,
        description="Burn subtitles into final video",
    )

    subtitle_mode: Literal["original", "translated", "both"] = Field(
        default="translated",
        description="Subtitle text mode: original, translated, or both",
    )

    subtitle_merge_gap: float = Field(
        default=0.35,
        description="Maximum gap in seconds to merge adjacent subtitle cues",
    )

    subtitle_max_lines: int = Field(
        default=2,
        description="Maximum lines per merged subtitle cue",
    )

    subtitle_max_chars: int = Field(
        default=42,
        description="Maximum characters before splitting a subtitle cue",
    )

    subtitle_max_duration: float = Field(
        default=6.0,
        description="Maximum duration in seconds for a split subtitle cue",
    )
    
    # ==================== Paths ====================
    
    @property
    def model_cache_path(self) -> Path:
        """Get the model cache directory path."""
        return Path(self.model_cache_dir)
    
    @property
    def output_path(self) -> Path:
        """Get the output directory path."""
        return Path(self.output_dir)
    
    @property
    def temp_path(self) -> Path:
        """Get the temporary directory path."""
        return Path(self.temp_dir)
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [
            self.model_cache_path,
            self.output_path,
            self.temp_path,
            self.audio_output_path,
            self.video_output_path,
            self.subtitle_output_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    # Subdirectories
    @property
    def audio_output_path(self) -> Path:
        """Get the audio output directory path."""
        return self.output_path / "audio"
    
    @property
    def video_output_path(self) -> Path:
        """Get the video output directory path."""
        return self.output_path / "video"
    
    @property
    def subtitle_output_path(self) -> Path:
        """Get the subtitle output directory path."""
        return self.output_path / "subtitles"


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
        _config.ensure_directories()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
