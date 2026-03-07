"""Tests for audio processing module."""

import pytest
from pathlib import Path

from video_translator.processing.audio import AudioProcessor, AudioInfo


class TestAudioProcessor:
    """Test AudioProcessor class."""
    
    def test_init_default(self):
        """Test default initialization."""
        processor = AudioProcessor()
        
        assert processor.ffmpeg_path == "ffmpeg"
        assert processor.sample_rate == 16000
        assert processor.channels == 1
    
    def test_init_custom(self):
        """Test custom initialization."""
        processor = AudioProcessor(
            ffmpeg_path="/usr/bin/ffmpeg",
            sample_rate=24000,
            channels=2,
        )
        
        assert processor.ffmpeg_path == "/usr/bin/ffmpeg"
        assert processor.sample_rate == 24000
        assert processor.channels == 2
    
    def test_verify_ffmpeg_success(self, monkeypatch):
        """Test FFmpeg verification succeeds."""
        # Mock subprocess to simulate successful FFmpeg check
        class MockResult:
            returncode = 0
            stdout = "ffmpeg version 5.0 Copyright (c) 2000-2023"
            stderr = ""
        
        def mock_run(*args, **kwargs):
            return MockResult()
        
        monkeypatch.setattr("subprocess.run", mock_run)
        
        # Should not raise
        processor = AudioProcessor()
        assert processor is not None
    
    def test_verify_ffmpeg_failure(self, monkeypatch):
        """Test FFmpeg verification fails."""
        import subprocess
        
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("ffmpeg not found")
        
        monkeypatch.setattr("subprocess.run", mock_run)
        
        with pytest.raises(RuntimeError, match="FFmpeg not found"):
            AudioProcessor()


class TestAudioInfo:
    """Test AudioInfo dataclass."""
    
    def test_audio_info_creation(self):
        """Test creating AudioInfo instance."""
        info = AudioInfo(
            path=Path("/test/audio.wav"),
            duration=10.5,
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
            bitrate=256000,
        )
        
        assert info.path == Path("/test/audio.wav")
        assert info.duration == 10.5
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.codec == "pcm_s16le"
        assert info.bitrate == 256000
