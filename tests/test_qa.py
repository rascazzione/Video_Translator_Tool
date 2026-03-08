"""Tests for segment QA checks."""

from pathlib import Path

import numpy as np
import soundfile as sf

from video_translator.processing.qa import SegmentQA


def test_detects_overlap_and_duration_mismatch(tmp_path: Path):
    """QA should flag timing regressions."""
    qa = SegmentQA(max_duration_error_ratio=0.1, max_gap_seconds=0.5)
    audio_path = tmp_path / "seg.wav"
    sf.write(str(audio_path), np.zeros(16000, dtype=np.float32), 16000)

    segments = [
        {"start": 0.0, "end": 1.0, "actual_duration": 1.0, "audio_path": str(audio_path)},
        {"start": 0.8, "end": 2.0, "actual_duration": 0.6, "audio_path": str(audio_path)},
    ]

    issues = qa.validate(segments)
    issue_types = {issue.issue_type for issue in issues}

    assert "overlap" in issue_types
    assert "duration_mismatch" in issue_types


def test_detects_clipping(tmp_path: Path):
    """QA should flag clipped waveforms."""
    qa = SegmentQA(clip_threshold=0.95)
    clipped_path = tmp_path / "clipped.wav"
    clipped = np.ones(16000, dtype=np.float32)
    sf.write(str(clipped_path), clipped, 16000)

    issues = qa.validate(
        [
            {
                "start": 0.0,
                "end": 1.0,
                "actual_duration": 1.0,
                "audio_path": str(clipped_path),
            }
        ]
    )

    assert any(issue.issue_type == "clipping" for issue in issues)
