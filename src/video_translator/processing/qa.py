"""Quality checks for segment-level dubbing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf


@dataclass
class QAIssue:
    """Validation issue found during QA."""

    segment_index: int
    issue_type: str
    severity: str
    message: str


class SegmentQA:
    """Rule-based QA checks for timeline and synthesized segments."""

    def __init__(
        self,
        max_duration_error_ratio: float = 0.2,
        max_gap_seconds: float = 2.0,
        max_overlap_seconds: float = 0.05,
        clip_threshold: float = 0.99,
    ):
        self.max_duration_error_ratio = max_duration_error_ratio
        self.max_gap_seconds = max_gap_seconds
        self.max_overlap_seconds = max_overlap_seconds
        self.clip_threshold = clip_threshold

    def validate(self, segments: List[Dict[str, Any]]) -> List[QAIssue]:
        """Validate per-segment timing and waveform quality."""
        issues: List[QAIssue] = []
        if not segments:
            return issues

        ordered = sorted(segments, key=lambda x: float(x.get("start", 0.0)))
        prev_end = float(ordered[0].get("start", 0.0))

        for idx, seg in enumerate(ordered):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            target_duration = max(0.0, end - start)
            actual_duration = float(seg.get("actual_duration", 0.0))

            if target_duration > 0 and actual_duration > 0:
                err_ratio = abs(actual_duration - target_duration) / target_duration
                if err_ratio > self.max_duration_error_ratio:
                    issues.append(
                        QAIssue(
                            segment_index=idx,
                            issue_type="duration_mismatch",
                            severity="medium",
                            message=(
                                f"target={target_duration:.3f}s actual={actual_duration:.3f}s "
                                f"error={err_ratio:.2%}"
                            ),
                        )
                    )

            gap = start - prev_end
            if gap > self.max_gap_seconds:
                issues.append(
                    QAIssue(
                        segment_index=idx,
                        issue_type="excessive_gap",
                        severity="low",
                        message=f"Gap of {gap:.3f}s before segment",
                    )
                )
            if gap < -self.max_overlap_seconds:
                issues.append(
                    QAIssue(
                        segment_index=idx,
                        issue_type="overlap",
                        severity="high",
                        message=f"Overlap of {abs(gap):.3f}s with previous segment",
                    )
                )

            audio_path = seg.get("audio_path")
            if audio_path:
                clipped = self._is_clipped(Path(audio_path))
                if clipped:
                    issues.append(
                        QAIssue(
                            segment_index=idx,
                            issue_type="clipping",
                            severity="medium",
                            message=f"Waveform peak exceeds {self.clip_threshold:.2f}",
                        )
                    )

            prev_end = max(prev_end, end)

        return issues

    def _is_clipped(self, audio_path: Path) -> bool:
        """Check if a waveform is clipped."""
        if not audio_path.exists():
            return False

        audio, _ = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if audio.size == 0:
            return False

        peak = float(np.max(np.abs(audio)))
        return peak >= self.clip_threshold


def run_segment_qa(segments: List[Dict[str, Any]]) -> List[QAIssue]:
    """Convenience helper for segment QA."""
    qa = SegmentQA()
    return qa.validate(segments)
