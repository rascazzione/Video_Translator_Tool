"""Tests for subtitle generation module."""

import pytest
from pathlib import Path
import tempfile

from video_translator.processing.subtitles import (
    SubtitleGenerator,
    SubtitleSegment,
    generate_srt,
    generate_vtt,
)


class TestSubtitleGenerator:
    """Test SubtitleGenerator class."""
    
    def test_seconds_to_srt_time(self):
        """Test SRT time conversion."""
        generator = SubtitleGenerator()
        
        assert generator._seconds_to_srt_time(0) == "00:00:00,000"
        assert generator._seconds_to_srt_time(1.5) == "00:00:01,500"
        assert generator._seconds_to_srt_time(61.234) == "00:01:01,234"
        assert generator._seconds_to_srt_time(3661.999) in {"01:01:01,998", "01:01:01,999"}
    
    def test_seconds_to_vtt_time(self):
        """Test VTT time conversion."""
        generator = SubtitleGenerator()
        
        assert generator._seconds_to_vtt_time(0) == "00:00:00.000"
        assert generator._seconds_to_vtt_time(1.5) == "00:00:01.500"
        assert generator._seconds_to_vtt_time(61.234) == "00:01:01.234"
    
    def test_generate_srt(self, tmp_path):
        """Test SRT file generation."""
        generator = SubtitleGenerator()
        output_path = tmp_path / "test.srt"
        
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 3.0, "end": 5.5, "text": "This is a test"},
        ]
        
        result_path = generator.generate_srt(segments, output_path)
        
        assert result_path == output_path
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "1" in content
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "Hello world" in content
        assert "2" in content
        assert "00:00:03,000 --> 00:00:05,500" in content
        assert "This is a test" in content
    
    def test_generate_vtt(self, tmp_path):
        """Test VTT file generation."""
        generator = SubtitleGenerator()
        output_path = tmp_path / "test.vtt"
        
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
        ]
        
        result_path = generator.generate_vtt(segments, output_path)
        
        assert result_path == output_path
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "WEBVTT" in content
        assert "00:00:00.000 --> 00:00:02.500" in content
        assert "Hello world" in content
    
    def test_merge_segments(self):
        """Test merging nearby segments."""
        generator = SubtitleGenerator()
        
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.2, "end": 2.0, "text": "world"},
            {"start": 3.0, "end": 4.0, "text": "Test"},
        ]
        
        merged = generator.merge_segments(segments, max_gap=0.5)
        
        # First two should merge, third stays separate
        assert len(merged) == 2
        assert merged[0]["text"] == "Hello\nworld"
        assert merged[1]["text"] == "Test"
    
    def test_split_long_segment(self):
        """Test splitting long segments."""
        generator = SubtitleGenerator()
        
        segment = {
            "start": 0.0,
            "end": 10.0,
            "text": "This is a very long subtitle that should be split into multiple segments because it exceeds the maximum character limit",
        }
        
        split = generator.split_long_segment(segment, max_chars=30)
        
        assert len(split) > 1
        for seg in split:
            assert len(seg["text"]) <= 35  # Allow some margin


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_generate_srt_function(self, tmp_path):
        """Test generate_srt convenience function."""
        segments = [{"start": 0, "end": 1, "text": "Test"}]
        output_path = tmp_path / "test.srt"
        
        result = generate_srt(segments, output_path)
        
        assert result.exists()
        assert "Test" in result.read_text()
    
    def test_generate_vtt_function(self, tmp_path):
        """Test generate_vtt convenience function."""
        segments = [{"start": 0, "end": 1, "text": "Test"}]
        output_path = tmp_path / "test.vtt"
        
        result = generate_vtt(segments, output_path)
        
        assert result.exists()
        assert "Test" in result.read_text()
