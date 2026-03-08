"""Video processing utilities using FFmpeg."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Information about a video file."""
    
    path: Path
    """Path to the video file."""
    
    duration: float
    """Duration in seconds."""
    
    width: int
    """Video width in pixels."""
    
    height: int
    """Video height in pixels."""
    
    fps: float
    """Frames per second."""
    
    video_codec: str
    """Video codec."""
    
    audio_codec: Optional[str]
    """Audio codec (None if no audio)."""


class VideoProcessor:
    """Video processing utilities using FFmpeg."""
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """Initialize video processor.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable (uses system FFmpeg if None).
        """
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
    
    def mux_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        video_codec: str = "copy",
        audio_codec: str = "aac",
        shortest: bool = True,
        keep_original_audio: bool = False,
    ) -> VideoInfo:
        """Mux (combine) audio with video.
        
        Args:
            video_path: Path to input video file.
            audio_path: Path to input audio file.
            output_path: Path for output video file.
            video_codec: Video codec ('copy' for no re-encoding).
            audio_codec: Audio codec ('aac', 'mp3', 'copy', etc.).
            shortest: If True, output duration matches shorter input.
            keep_original_audio: If True, keeps original audio as secondary track.
        
        Returns:
            VideoInfo with details about output video.
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command
        if keep_original_audio:
            # Keep original audio as secondary track
            cmd = [
                self.ffmpeg_path,
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", video_codec,
                "-c:a:0", audio_codec,
                "-c:a:1", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-map", "0:a:0",
                "-disposition:a:0", "default",
                "-disposition:a:1", "0",
            ]
        else:
            # Replace audio completely
            cmd = [
                self.ffmpeg_path,
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", video_codec,
                "-c:a", audio_codec,
                "-map", "0:v:0",
                "-map", "1:a:0",
            ]
        
        if shortest:
            cmd.append("-shortest")
        
        cmd.extend(["-y", str(output_path)])
        
        logger.info(f"Muxing audio with video: {output_path.name}...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return self.get_video_info(output_path)
    
    def replace_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        audio_codec: str = "aac",
        audio_delay: float = 0.0,
    ) -> VideoInfo:
        """Replace audio track in video with new audio.
        
        This is a convenience wrapper around mux_audio_video that removes
        all original audio tracks.
        
        Args:
            video_path: Path to input video file.
            audio_path: Path to new audio file.
            output_path: Path for output video file.
            audio_codec: Audio codec for output.
            audio_delay: Delay in seconds before starting the new audio.
        
        Returns:
            VideoInfo with details about output video.
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        
        # Get video duration to pad audio if needed
        video_info = self.get_video_info(video_path)
        video_duration = video_info.duration
        
        logger.info(f"Replacing audio in {video_path.name}...")
        logger.info(f"Video duration: {video_duration}s")
        logger.info(f"Audio delay: {audio_delay}s")
        
        # Create temp directory for padded audio
        import tempfile
        import os
        
        # Pad audio with delay at start and padding at end to match video duration
        delayed_audio = audio_path.parent / f"delayed_{audio_path.name}"
        
        # Build filter: add delay at start, then pad to match video duration
        # adelay adds delay in milliseconds
        delay_ms = int(audio_delay * 1000)
        total_duration = video_duration
        
        # Use afilter to add delay at start and pad to match video duration
        filter_cmd = f"adelay={delay_ms}|{delay_ms},apad=whole_dur={total_duration}"
        
        pad_cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(audio_path),
            "-af", filter_cmd,
            str(delayed_audio),
        ]
        
        result = subprocess.run(pad_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Audio delay/padding failed: {result.stderr}")
            # Fall back to original audio
            delayed_audio = audio_path
        
        # Use -map to select all streams except audio from video
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video_path),
            "-i", str(delayed_audio),
            "-c:v", "copy",
            "-c:a", audio_codec,
            "-map", "0",  # All streams from video
            "-map", "-0:a",  # Exclude audio from video
            "-map", "1:a",  # Add new (delayed/padded) audio
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return self.get_video_info(output_path)

    def burn_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
    ) -> VideoInfo:
        """Burn subtitles into the video (hard subtitles)."""
        video_path = Path(video_path)
        subtitle_path = Path(subtitle_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not subtitle_path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        subtitle_filter_path = (
            subtitle_path.resolve().as_posix().replace("\\", "/").replace(":", r"\:")
        )
        subtitle_filter_path = subtitle_filter_path.replace("'", r"\'")
        subtitle_filter = f"subtitles='{subtitle_filter_path}'"

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video_path),
            "-vf", subtitle_filter,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path),
        ]

        logger.info("Burning subtitles into video: %s", output_path.name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg subtitle burn failed: {result.stderr}")

        return self.get_video_info(output_path)
    
    def get_video_info(self, video_path: Path) -> VideoInfo:
        """Get information about a video file using FFprobe."""
        ffprobe_path = self.ffmpeg_path.replace("ffmpeg", "ffprobe")
        
        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        import json
        data = json.loads(result.stdout)
        
        # Find video and audio streams
        video_stream = None
        audio_stream = None
        
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream
        
        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")
        
        format_info = data.get("format", {})
        
        # Parse FPS
        fps_str = video_stream.get("r_frame_rate", "0/1")
        try:
            num, denom = map(int, fps_str.split("/"))
            fps = num / denom if denom > 0 else 0
        except (ValueError, ZeroDivisionError):
            fps = 0
        
        return VideoInfo(
            path=video_path,
            duration=float(format_info.get("duration", 0)),
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=fps,
            video_codec=video_stream.get("codec_name", "unknown"),
            audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        )
    
    def extract_frame(
        self,
        video_path: Path,
        output_path: Path,
        timestamp: float = 0.0,
    ) -> Path:
        """Extract a single frame from video at specified timestamp.
        
        Args:
            video_path: Path to input video file.
            output_path: Path for output image file.
            timestamp: Timestamp in seconds to extract frame.
        
        Returns:
            Path to extracted image.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        # Convert timestamp to HH:MM:SS format
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = timestamp % 60
        timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        
        cmd = [
            self.ffmpeg_path,
            "-ss", timestamp_str,
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        return output_path
    
    def trim_video(
        self,
        video_path: Path,
        output_path: Path,
        start: float = 0.0,
        duration: Optional[float] = None,
        end: Optional[float] = None,
    ) -> VideoInfo:
        """Trim video to specified time range.
        
        Args:
            video_path: Path to input video file.
            output_path: Path for output video file.
            start: Start time in seconds.
            duration: Duration in seconds (alternative to end).
            end: End time in seconds (alternative to duration).
        
        Returns:
            VideoInfo with details about trimmed video.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if duration is None and end is None:
            raise ValueError("Either duration or end must be specified")
        
        if end is not None and duration is not None:
            raise ValueError("Specify either duration or end, not both")
        
        cmd = [
            self.ffmpeg_path,
            "-ss", str(start),
            "-i", str(video_path),
        ]
        
        if duration is not None:
            cmd.extend(["-t", str(duration)])
        else:
            cmd.extend(["-to", str(end)])
        
        cmd.extend([
            "-c:v", "copy",
            "-c:a", "copy",
            "-y",
            str(output_path),
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        return self.get_video_info(output_path)


def mux_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    shortest: bool = True,
) -> VideoInfo:
    """Convenience function to mux audio with video.
    
    Args:
        video_path: Path to input video file.
        audio_path: Path to input audio file.
        output_path: Path for output video file.
        video_codec: Video codec.
        audio_codec: Audio codec.
        shortest: Match shorter input duration.
    
    Returns:
        VideoInfo with details about output video.
    """
    processor = VideoProcessor()
    return processor.mux_audio_video(
        video_path, audio_path, output_path,
        video_codec=video_codec,
        audio_codec=audio_codec,
        shortest=shortest,
    )
