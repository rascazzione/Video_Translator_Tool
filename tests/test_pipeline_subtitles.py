"""Tests for subtitle shaping and translation status helpers."""

from pathlib import Path
from types import SimpleNamespace

from video_translator.config import Config
from video_translator.pipeline import (
    SegmentTranslationResult,
    VideoTranslator,
)
from video_translator.processing.subtitles import SubtitleGenerator


def _translator_for_subtitles() -> VideoTranslator:
    translator = VideoTranslator.__new__(VideoTranslator)
    translator.config = SimpleNamespace(
        subtitle_merge_gap=0.35,
        subtitle_max_lines=2,
        subtitle_max_chars=42,
        subtitle_max_duration=6.0,
    )
    translator._subtitle_generator = SubtitleGenerator()
    return translator


def test_translated_subtitles_merge_adjacent_segments():
    """Translated subtitle cues should merge when the gap is small."""
    translator = _translator_for_subtitles()
    segments = [
        SegmentTranslationResult(
            segment_id=0,
            start=0.0,
            end=1.0,
            source_text="Hello",
            translated_text="Hola",
            audio_path=Path("/tmp/seg0.wav"),
            actual_duration=1.0,
        ),
        SegmentTranslationResult(
            segment_id=1,
            start=1.2,
            end=2.0,
            source_text="world",
            translated_text="mundo",
            audio_path=Path("/tmp/seg1.wav"),
            actual_duration=0.8,
        ),
    ]

    subtitle_segments = translator._build_subtitle_segments(segments, mode="translated")

    assert len(subtitle_segments) == 1
    assert subtitle_segments[0]["text"] == "Hola\nmundo"


def test_bilingual_subtitles_keep_one_cue_per_segment():
    """Dual-language subtitles should avoid post-merge stacking."""
    translator = _translator_for_subtitles()
    segments = [
        SegmentTranslationResult(
            segment_id=0,
            start=0.0,
            end=1.0,
            source_text="Hello",
            translated_text="Hola",
            audio_path=Path("/tmp/seg0.wav"),
            actual_duration=1.0,
        ),
        SegmentTranslationResult(
            segment_id=1,
            start=1.2,
            end=2.0,
            source_text="world",
            translated_text="mundo",
            audio_path=Path("/tmp/seg1.wav"),
            actual_duration=0.8,
        ),
    ]

    subtitle_segments = translator._build_subtitle_segments(segments, mode="both")

    assert len(subtitle_segments) == 2
    assert subtitle_segments[0]["text"] == "Hello\nHola"
    assert subtitle_segments[1]["text"] == "world\nmundo"


def test_translate_text_result_skips_same_language():
    """Translation metadata should expose same-language no-op behavior."""
    translator = VideoTranslator(config=Config())

    result = translator._translate_text_result(
        text="hello world",
        source_language="en",
        target_language="English",
    )

    assert result.text == "hello world"
    assert result.status == "skipped_same_language"


def test_translate_text_result_marks_fallback(monkeypatch):
    """Translation metadata should record fallback-to-source behavior."""
    translator = VideoTranslator(config=Config())
    monkeypatch.setattr(
        translator,
        "_get_translation_backend",
        lambda source_code: (_ for _ in ()).throw(RuntimeError("backend failed")),
    )

    result = translator._translate_text_result(
        text="hello world",
        source_language="en",
        target_language="es",
    )

    assert result.text == "hello world"
    assert result.status == "fallback_original"
    assert result.error == "backend failed"
