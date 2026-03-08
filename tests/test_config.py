"""Tests for configuration module."""

import os
from pathlib import Path

import pytest

from video_translator.config import Config, get_config, set_config


def test_default_config():
    """Test default configuration values."""
    config = Config()
    
    assert config.qwen_asr_model == "Qwen/Qwen3-ASR-1.7B"
    assert config.qwen_tts_model == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert config.qwen_aligner_model == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert config.device == "auto"
    assert config.precision == "bf16"
    assert config.flash_attention is True
    assert config.use_vad is True
    assert config.keep_background_audio is False
    assert config.background_audio_volume == 0.2
    assert config.embed_subtitles is False
    assert config.subtitle_mode == "translated"
    assert config.api_port == 8000


def test_config_from_env():
    """Test configuration from environment variables."""
    # Set environment variables
    os.environ["QWEN_ASR_MODEL"] = "Qwen/Qwen3-ASR-0.6B"
    os.environ["DEVICE"] = "cpu"
    os.environ["API_PORT"] = "9000"
    
    config = Config()
    
    assert config.qwen_asr_model == "Qwen/Qwen3-ASR-0.6B"
    assert config.device == "cpu"
    assert config.api_port == 9000
    
    # Clean up
    del os.environ["QWEN_ASR_MODEL"]
    del os.environ["DEVICE"]
    del os.environ["API_PORT"]


def test_config_ensure_directories(tmp_path):
    """Test that config creates required directories."""
    config = Config(
        model_cache_dir=str(tmp_path / "models"),
        output_dir=str(tmp_path / "output"),
        temp_dir=str(tmp_path / "temp"),
    )
    config.ensure_directories()
    
    assert config.model_cache_path.exists()
    assert config.output_path.exists()
    assert config.temp_path.exists()
    
    # Check subdirectories
    assert config.audio_output_path.exists()
    assert config.video_output_path.exists()
    assert config.subtitle_output_path.exists()


def test_get_config_singleton():
    """Test that get_config returns a singleton."""
    config1 = get_config()
    config2 = get_config()
    
    assert config1 is config2


def test_set_config():
    """Test setting custom config."""
    custom_config = Config(device="cpu")
    set_config(custom_config)
    
    assert get_config() is custom_config
