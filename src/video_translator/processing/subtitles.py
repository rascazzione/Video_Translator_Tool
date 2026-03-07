"""Subtitle generation utilities."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class SubtitleSegment:
    """A single subtitle segment."""
    
    index: int
    """Segment index (1-based)."""
    
    start: float
    """Start time in seconds."""
    
    end: float
    """End time in seconds."""
    
    text: str
    """Subtitle text."""


class SubtitleGenerator:
    """Generate subtitle files in various formats."""
    
    def __init__(self):
        """Initialize subtitle generator."""
        pass
    
    def generate_srt(
        self,
        segments: List[Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """Generate SRT subtitle file.
        
        Args:
            segments: List of segments with 'start', 'end', and 'text' keys.
            output_path: Path for output SRT file.
        
        Returns:
            Path to generated SRT file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._seconds_to_srt_time(seg["start"])
            end = self._seconds_to_srt_time(seg["end"])
            text = seg.get("text", "")
            
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")  # Empty line between segments
        
        content = "\n".join(lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Generated SRT file: {output_path}")
        return output_path
    
    def generate_vtt(
        self,
        segments: List[Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """Generate WebVTT subtitle file.
        
        Args:
            segments: List of segments with 'start', 'end', and 'text' keys.
            output_path: Path for output VTT file.
        
        Returns:
            Path to generated VTT file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = ["WEBVTT", ""]
        
        for seg in segments:
            start = self._seconds_to_vtt_time(seg["start"])
            end = self._seconds_to_vtt_time(seg["end"])
            text = seg.get("text", "")
            
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        
        content = "\n".join(lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Generated VTT file: {output_path}")
        return output_path
    
    def generate_json(
        self,
        segments: List[Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """Generate JSON subtitle file.
        
        Args:
            segments: List of segments with 'start', 'end', and 'text' keys.
            output_path: Path for output JSON file.
        
        Returns:
            Path to generated JSON file.
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated JSON subtitle file: {output_path}")
        return output_path
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def merge_segments(
        self,
        segments: List[Dict[str, Any]],
        max_gap: float = 0.5,
        max_lines: int = 2,
    ) -> List[Dict[str, Any]]:
        """Merge nearby subtitle segments.
        
        Args:
            segments: List of segments to merge.
            max_gap: Maximum gap in seconds to merge.
            max_lines: Maximum lines per subtitle.
        
        Returns:
            List of merged segments.
        """
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        current_lines = [current["text"]]
        
        for seg in segments[1:]:
            gap = seg["start"] - current["end"]
            
            if gap <= max_gap and len(current_lines) < max_lines:
                # Merge with current segment
                current["end"] = seg["end"]
                current_lines.append(seg["text"])
                current["text"] = "\n".join(current_lines)
            else:
                # Start new segment
                merged.append(current)
                current = seg.copy()
                current_lines = [seg["text"]]
        
        merged.append(current)
        return merged
    
    def split_long_segment(
        self,
        segment: Dict[str, Any],
        max_chars: int = 42,
        max_duration: float = 6.0,
    ) -> List[Dict[str, Any]]:
        """Split a long subtitle segment into multiple segments.
        
        Args:
            segment: Segment to split.
            max_chars: Maximum characters per line.
            max_duration: Maximum duration per segment in seconds.
        
        Returns:
            List of split segments.
        """
        text = segment.get("text", "")
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        
        if len(text) <= max_chars:
            return [segment]
        
        # Split text into words
        words = text.split()
        segments = []
        
        current_text = []
        current_start = start
        duration_per_word = (end - start) / len(words) if words else 0
        
        for word in words:
            if len(" ".join(current_text + [word])) <= max_chars:
                current_text.append(word)
            else:
                # Create segment
                if current_text:
                    current_end = current_start + (len(current_text) * duration_per_word)
                    segments.append({
                        "start": current_start,
                        "end": min(current_end, current_start + max_duration),
                        "text": " ".join(current_text),
                    })
                    current_start = current_end
                
                current_text = [word]
        
        # Add remaining text
        if current_text:
            segments.append({
                "start": current_start,
                "end": end,
                "text": " ".join(current_text),
            })
        
        return segments


def generate_srt(
    segments: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Convenience function to generate SRT file.
    
    Args:
        segments: List of segments with 'start', 'end', and 'text' keys.
        output_path: Path for output SRT file.
    
    Returns:
        Path to generated SRT file.
    """
    generator = SubtitleGenerator()
    return generator.generate_srt(segments, output_path)


def generate_vtt(
    segments: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Convenience function to generate VTT file.
    
    Args:
        segments: List of segments with 'start', 'end', and 'text' keys.
        output_path: Path for output VTT file.
    
    Returns:
        Path to generated VTT file.
    """
    generator = SubtitleGenerator()
    return generator.generate_vtt(segments, output_path)
