"""Voice activity detection utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeechRegion:
    """A detected speech region."""

    start: float
    """Start time in seconds."""

    end: float
    """End time in seconds."""

    confidence: float = 1.0
    """Region confidence score in the range [0, 1]."""


class SileroVAD:
    """Voice activity detector using Silero VAD with a safe fallback."""

    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 200,
    ):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self._model = None

    def _load_model(self) -> bool:
        """Load Silero model if available."""
        if self._model is not None:
            return True

        try:
            from silero_vad import load_silero_vad
        except ImportError:
            logger.warning(
                "silero-vad is not installed; falling back to energy-based VAD."
            )
            return False

        self._model = load_silero_vad()
        return True

    def detect(self, audio_path: Path) -> List[SpeechRegion]:
        """Detect speech regions from an audio file."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._load_model():
            regions = self._detect_with_silero(audio_path)
            if regions:
                return regions

        return self._detect_energy_based(audio_path)

    def _detect_with_silero(self, audio_path: Path) -> List[SpeechRegion]:
        """Run VAD using Silero package APIs."""
        try:
            from silero_vad import get_speech_timestamps, read_audio
        except ImportError:
            return []

        try:
            audio = read_audio(str(audio_path), sampling_rate=self.sampling_rate)
            timestamps = get_speech_timestamps(
                audio=audio,
                model=self._model,
                sampling_rate=self.sampling_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
            )
        except Exception as exc:  # pragma: no cover - backend-specific
            logger.warning("Silero VAD failed, using fallback VAD: %s", exc)
            return []

        regions: List[SpeechRegion] = []
        for item in timestamps:
            # Silero returns sample indices.
            start = float(item.get("start", 0)) / self.sampling_rate
            end = float(item.get("end", 0)) / self.sampling_rate
            if end <= start:
                continue
            regions.append(
                SpeechRegion(
                    start=start,
                    end=end,
                    confidence=float(item.get("confidence", 1.0)),
                )
            )
        return regions

    def _detect_energy_based(self, audio_path: Path) -> List[SpeechRegion]:
        """Energy-based fallback for environments without Silero."""
        try:
            import librosa

            audio, sample_rate = librosa.load(
                str(audio_path), sr=self.sampling_rate, mono=True
            )
        except Exception:
            import soundfile as sf

            audio, sample_rate = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sample_rate != self.sampling_rate:
                import librosa

                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=self.sampling_rate
                )
                sample_rate = self.sampling_rate

        if audio.size == 0:
            return []

        frame_length = int(0.03 * sample_rate)
        hop_length = int(0.01 * sample_rate)

        # RMS envelope used as simple speech proxy.
        padded = np.pad(audio, (0, max(0, frame_length - 1)))
        frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[
            ::hop_length
        ]
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        if rms.size == 0:
            return []

        adaptive_threshold = max(self.threshold * np.max(rms), np.mean(rms) * 1.25)
        speech_mask = rms >= adaptive_threshold

        min_speech = self.min_speech_duration_ms / 1000.0
        min_silence = self.min_silence_duration_ms / 1000.0

        regions: List[SpeechRegion] = []
        in_speech = False
        start_time = 0.0
        for idx, is_speech in enumerate(speech_mask):
            current_time = idx * hop_length / sample_rate
            if is_speech and not in_speech:
                in_speech = True
                start_time = current_time
            elif not is_speech and in_speech:
                in_speech = False
                end_time = current_time
                if end_time - start_time >= min_speech:
                    regions.append(
                        SpeechRegion(start=start_time, end=end_time, confidence=0.5)
                    )

        if in_speech:
            end_time = len(audio) / sample_rate
            if end_time - start_time >= min_speech:
                regions.append(SpeechRegion(start=start_time, end=end_time, confidence=0.5))

        # Merge tiny silence gaps.
        merged: List[SpeechRegion] = []
        for region in regions:
            if not merged:
                merged.append(region)
                continue
            prev = merged[-1]
            gap = region.start - prev.end
            if gap <= min_silence:
                prev.end = region.end
                prev.confidence = max(prev.confidence, region.confidence)
            else:
                merged.append(region)
        return merged


def detect_speech_regions(
    audio_path: Path,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 200,
) -> List[SpeechRegion]:
    """Convenience wrapper for VAD detection."""
    detector = SileroVAD(
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    return detector.detect(audio_path)
