"""Tests for VAD processing."""

from pathlib import Path

import numpy as np
import soundfile as sf

from video_translator.processing.vad import SileroVAD, SpeechRegion


def test_detect_fallback_path(monkeypatch, tmp_path: Path):
    """Detector should return fallback regions when Silero is unavailable."""
    detector = SileroVAD()

    expected = [SpeechRegion(start=0.1, end=0.9, confidence=0.5)]
    monkeypatch.setattr(detector, "_load_model", lambda: False)
    monkeypatch.setattr(detector, "_detect_energy_based", lambda _: expected)

    audio_path = tmp_path / "audio.wav"
    sf.write(str(audio_path), np.zeros(1600, dtype=np.float32), 16000)

    regions = detector.detect(audio_path)
    assert regions == expected


def test_energy_based_detects_speech_region(tmp_path: Path):
    """Fallback detector should identify an obvious speech burst."""
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    speech = np.random.uniform(-0.6, 0.6, sr).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])

    audio_path = tmp_path / "speech.wav"
    sf.write(str(audio_path), audio, sr)

    detector = SileroVAD(threshold=0.1, sampling_rate=sr)
    regions = detector._detect_energy_based(audio_path)

    assert len(regions) >= 1
    assert any(region.start < 1.5 and region.end > 1.5 for region in regions)
