"""Tests for pipeline region-building behavior."""

from types import SimpleNamespace

from video_translator.pipeline import VideoTranslator
from video_translator.processing.vad import SpeechRegion


def _translator_for_regions(
    *,
    max_segment_duration: float = 30.0,
    min_segment_duration: float = 0.4,
    vad_min_silence_duration_ms: int = 200,
) -> VideoTranslator:
    translator = VideoTranslator.__new__(VideoTranslator)
    translator.config = SimpleNamespace(
        max_segment_duration=max_segment_duration,
        min_segment_duration=min_segment_duration,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
    )
    return translator


def test_no_vad_regions_falls_back_to_fixed_chunks():
    """When VAD is off/unavailable, timeline is chunked by max segment length."""
    translator = _translator_for_regions(max_segment_duration=30.0)

    regions = translator._build_processing_regions([], total_duration=65.0)

    assert len(regions) == 3
    assert regions[0] == SpeechRegion(start=0.0, end=30.0, confidence=1.0)
    assert regions[1] == SpeechRegion(start=30.0, end=60.0, confidence=1.0)
    assert regions[2] == SpeechRegion(start=60.0, end=65.0, confidence=1.0)


def test_tiny_vad_regions_fall_back_to_fixed_chunks():
    """If VAD output is unusably short, fallback chunks prevent zero work segments."""
    translator = _translator_for_regions(max_segment_duration=30.0, min_segment_duration=0.4)
    vad_regions = [
        SpeechRegion(start=10.0, end=10.2, confidence=0.8),
        SpeechRegion(start=20.0, end=20.2, confidence=0.8),
    ]

    regions = translator._build_processing_regions(vad_regions, total_duration=61.0)

    assert len(regions) == 3
    assert regions[0] == SpeechRegion(start=0.0, end=30.0, confidence=1.0)
    assert regions[1] == SpeechRegion(start=30.0, end=60.0, confidence=1.0)
    assert regions[2] == SpeechRegion(start=60.0, end=61.0, confidence=1.0)
