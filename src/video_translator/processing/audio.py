"""Audio processing utilities using FFmpeg."""

import math
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioInfo:
    """Information about an audio file."""
    
    path: Path
    """Path to the audio file."""
    
    duration: float
    """Duration in seconds."""
    
    sample_rate: int
    """Sample rate in Hz."""
    
    channels: int
    """Number of audio channels."""
    
    codec: str
    """Audio codec."""
    
    bitrate: int
    """Bitrate in bits per second."""


class AudioProcessor:
    """Audio processing utilities using FFmpeg."""
    
    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        """Initialize audio processor.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable (uses system FFmpeg if None).
            sample_rate: Default sample rate for processing.
            channels: Default number of channels (1=mono, 2=stereo).
        """
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        self.sample_rate = sample_rate
        self.channels = channels
        
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"FFmpeg version: {result.stdout.split()[2]}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg:\n"
                "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Windows: choco install ffmpeg or download from ffmpeg.org"
            ) from e
    
    def extract_audio(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        codec: str = "pcm_s16le",
        format: str = "wav",
    ) -> AudioInfo:
        """Extract audio from video file.
        
        Args:
            video_path: Path to input video file.
            output_path: Path for output audio file (auto-generated if None).
            sample_rate: Output sample rate (default: 16000).
            channels: Number of channels (default: 1 for mono).
            codec: Audio codec (default: pcm_s16le for WAV).
            format: Output format (default: wav).
        
        Returns:
            AudioInfo with details about extracted audio.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        sample_rate = sample_rate or self.sample_rate
        channels = channels or self.channels
        
        # Generate output path if not provided
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_audio.{format}"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", codec,
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-y",  # Overwrite output
            str(output_path),
        ]
        
        logger.info(f"Extracting audio from {video_path.name}...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        # Get audio info
        info = self.get_audio_info(output_path)
        logger.info(f"Audio extracted: {info.duration:.2f}s, {info.sample_rate}Hz")
        
        return info
    
    def get_audio_info(self, audio_path: Path) -> AudioInfo:
        """Get information about an audio file using FFprobe."""
        ffprobe_path = self.ffmpeg_path.replace("ffmpeg", "ffprobe")
        
        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(audio_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        import json
        data = json.loads(result.stdout)
        
        # Extract audio stream info
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break
        
        if not audio_stream:
            raise ValueError(f"No audio stream found in {audio_path}")
        
        format_info = data.get("format", {})
        
        return AudioInfo(
            path=audio_path,
            duration=float(format_info.get("duration", 0)),
            sample_rate=int(audio_stream.get("sample_rate", 0)),
            channels=int(audio_stream.get("channels", 0)),
            codec=audio_stream.get("codec_name", "unknown"),
            bitrate=int(format_info.get("bit_rate", 0)),
        )
    
    def resample(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int,
        channels: Optional[int] = None,
    ) -> AudioInfo:
        """Resample audio file to different sample rate.
        
        Args:
            input_path: Path to input audio file.
            output_path: Path for output audio file.
            sample_rate: Target sample rate.
            channels: Target number of channels (optional).
        
        Returns:
            AudioInfo with details about resampled audio.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        channels = channels or self.channels
        
        cmd = [
            self.ffmpeg_path,
            "-i", str(input_path),
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-y",
            str(output_path),
        ]
        
        logger.debug(f"Resampling audio to {sample_rate}Hz...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        return self.get_audio_info(output_path)
    
    def concatenate(
        self,
        audio_files: list[Path],
        output_path: Path,
    ) -> AudioInfo:
        """Concatenate multiple audio files.
        
        Args:
            audio_files: List of audio file paths to concatenate.
            output_path: Path for output audio file.
        
        Returns:
            AudioInfo with details about concatenated audio.
        """
        if not audio_files:
            raise ValueError("No audio files provided")
        
        output_path = Path(output_path)
        
        # Create file list for FFmpeg concat demuxer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for audio_file in audio_files:
                f.write(f"file '{audio_file.absolute()}'\n")
            list_file = f.name
        
        try:
            cmd = [
                self.ffmpeg_path,
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                "-y",
                str(output_path),
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return self.get_audio_info(output_path)
        finally:
            import os
            os.unlink(list_file)

    def extract_segment(
        self,
        input_path: Path,
        output_path: Path,
        start: float,
        end: float,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> AudioInfo:
        """Extract a time-bounded audio segment."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if end <= start:
            raise ValueError(f"Invalid segment bounds: start={start}, end={end}")

        sample_rate = sample_rate or self.sample_rate
        channels = channels or self.channels
        duration = end - start

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(input_path),
            "-t", f"{duration:.3f}",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg segment extraction failed: {result.stderr}")
        return self.get_audio_info(output_path)

    def time_stretch_to_duration(
        self,
        input_path: Path,
        output_path: Path,
        target_duration: float,
        max_stretch_ratio: float = 1.2,
    ) -> AudioInfo:
        """Retimes audio to fit a target duration using mild time-stretch."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if target_duration <= 0:
            raise ValueError("target_duration must be > 0")

        info = self.get_audio_info(input_path)
        if info.duration <= 0:
            raise ValueError("Input audio has zero duration")

        stretch_ratio = target_duration / info.duration
        if stretch_ratio <= 0:
            raise ValueError("Computed stretch ratio must be > 0")

        # Keep retiming mild to avoid artifacts.
        if stretch_ratio < 1 / max_stretch_ratio or stretch_ratio > max_stretch_ratio:
            raise ValueError(
                f"Stretch ratio {stretch_ratio:.3f} exceeds max {max_stretch_ratio:.3f}"
            )

        import librosa
        import soundfile as sf

        audio, sample_rate = librosa.load(str(input_path), sr=None, mono=True)
        if audio.size == 0:
            raise ValueError("Input audio is empty")

        # librosa time_stretch rate > 1 speeds up (shorter), < 1 slows down.
        rate = info.duration / target_duration
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        sf.write(str(output_path), stretched, sample_rate)
        return self.get_audio_info(output_path)

    def assemble_timeline(
        self,
        segments: List[Dict[str, Any]],
        output_path: Path,
        total_duration: float,
        sample_rate: Optional[int] = None,
    ) -> AudioInfo:
        """Assemble segment audio files into one timeline-aligned track."""
        if total_duration <= 0:
            raise ValueError("total_duration must be > 0")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = sample_rate or self.sample_rate

        timeline_size = int(math.ceil(total_duration * sample_rate)) + 1
        mix = np.zeros(timeline_size, dtype=np.float32)

        import librosa
        import soundfile as sf

        for seg in segments:
            seg_path = Path(seg["audio_path"])
            if not seg_path.exists():
                logger.warning("Skipping missing segment audio: %s", seg_path)
                continue

            start = max(0.0, float(seg.get("start", 0.0)))
            start_idx = int(start * sample_rate)
            if start_idx >= timeline_size:
                continue

            audio, sr = sf.read(str(seg_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

            end_idx = min(timeline_size, start_idx + len(audio))
            if end_idx <= start_idx:
                continue
            mix[start_idx:end_idx] += audio[: end_idx - start_idx]

        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 1.0:
            mix = mix / peak

        sf.write(str(output_path), mix, sample_rate)
        return self.get_audio_info(output_path)


def extract_audio(
    video_path: Path,
    output_path: Optional[Path] = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> AudioInfo:
    """Convenience function to extract audio from video.
    
    Args:
        video_path: Path to input video file.
        output_path: Path for output audio file (optional).
        sample_rate: Output sample rate.
        channels: Number of channels.
    
    Returns:
        AudioInfo with details about extracted audio.
    """
    processor = AudioProcessor(sample_rate=sample_rate, channels=channels)
    return processor.extract_audio(video_path, output_path)
