"""Command-line interface for Video Translator."""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .config import Config, set_config
from .pipeline import VideoTranslator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()
_config_file_override: Optional[Path] = None

# Create Typer app
app = typer.Typer(
    name="video-translator",
    help="Video translation pipeline using Qwen3 models",
    add_completion=False,
)


def _build_config() -> Config:
    """Build config, optionally loading overrides from a CLI-provided env file."""
    if _config_file_override is not None:
        return Config(_env_file=_config_file_override)
    return Config()


def _notify_completion() -> None:
    """Emit a terminal bell to signal long-running completion."""
    try:
        console.bell()
    except Exception:
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"[bold]Video Translator[/bold] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose output",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """Video Translator CLI - Translate videos using AI."""
    global _config_file_override

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if config_file and config_file.exists():
        _config_file_override = config_file
        logger.info("Loading config from %s", config_file)


@app.command("transcribe")
def transcribe(
    input_path: Path = typer.Argument(
        ...,
        help="Path to input video or audio file",
        exists=True,
    ),
    output_path: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Language code (auto-detected if not specified)",
    ),
    no_srt: bool = typer.Option(
        False,
        "--no-srt",
        help="Skip SRT subtitle generation",
    ),
    model: str = typer.Option(
        "1.7B",
        "--model",
        "-m",
        help="ASR model size (0.6B or 1.7B)",
    ),
) -> None:
    """Transcribe audio from video file."""
    config = _build_config()

    # Select model based on size
    if model == "0.6B":
        config.qwen_asr_model = "Qwen/Qwen3-ASR-0.6B"
    else:
        config.qwen_asr_model = "Qwen/Qwen3-ASR-1.7B"
    
    set_config(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Initializing...", total=None)
        translator = VideoTranslator(config=config)
    
    console.print(f"📝 Transcribing: [bold]{input_path.name}[/bold]")
    
    try:
        result = translator.transcribe(
            input_path,
            output_dir=output_path,
            generate_srt=not no_srt,
            language=language,
        )
        
        console.print("\n[green]✓ Transcription complete![/green]")
        console.print(f"\n📄 Language: {result.language}")
        console.print(f"📝 Transcript: {result.text[:200]}..." if len(result.text) > 200 else f"📝 Transcript: {result.text}")
        
        if result.srt_path:
            console.print(f"📹 Subtitles: {result.srt_path}")
        
        if result.audio_path:
            console.print(f"🎵 Audio: {result.audio_path}")
        
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("tts")
def text_to_speech(
    input_path: Path = typer.Argument(
        ...,
        help="Path to input text file",
        exists=True,
    ),
    output_path: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for audio file",
    ),
    language: str = typer.Option(
        "English",
        "--language",
        "-l",
        help="Target language",
    ),
    speaker: Optional[str] = typer.Option(
        None,
        "--speaker",
        "-s",
        help="Preset speaker name",
    ),
    voice_description: Optional[str] = typer.Option(
        None,
        "--voice-design",
        "-d",
        help="Voice description for voice design",
    ),
    reference_audio: Optional[Path] = typer.Option(
        None,
        "--reference",
        "-r",
        help="Reference audio for voice cloning",
    ),
) -> None:
    """Generate speech from text."""
    config = _build_config()
    set_config(config)

    # Read input text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.wav"
    
    console.print(f"🗣️ Generating speech from: [bold]{input_path.name}[/bold]")
    
    try:
        translator = VideoTranslator(config=config)
        
        result = translator.synthesize_speech(
            text=text,
            output_path=output_path,
            language=language,
            speaker=speaker,
            voice_clone=reference_audio is not None,
            reference_audio=reference_audio,
            voice_design=voice_description is not None,
            voice_description=voice_description,
        )
        
        console.print(f"\n[green]✓ Speech generated![/green]")
        console.print(f"🎵 Output: {output_path}")
        console.print(f"⏱️ Duration: {result.duration:.2f}s")
        
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("translate-video")
def translate_video(
    input_path: Path = typer.Argument(
        ...,
        help="Path to input video file",
        exists=True,
    ),
    target_language: str = typer.Argument(
        ...,
        help="Target language code (e.g., 'es', 'fr', 'de')",
    ),
    source_language: Optional[str] = typer.Option(
        None,
        "--source-language",
        help="Force source language (e.g., 'en', 'es', 'hi', 'English', 'Hindi')",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results",
    ),
    no_voice_clone: bool = typer.Option(
        False,
        "--no-voice-clone",
        help="Disable voice cloning (use preset speaker)",
    ),
    no_subtitles: bool = typer.Option(
        False,
        "--no-subtitles",
        help="Skip subtitle generation",
    ),
    disable_vad: bool = typer.Option(
        False,
        "--disable-vad",
        help="Disable VAD segmentation and process as one full segment",
    ),
    speaker: Optional[str] = typer.Option(
        None,
        "--speaker",
        "-s",
        help="Preset speaker (if not voice cloning)",
    ),
    asr_model: str = typer.Option(
        "1.7B",
        "--asr-model",
        help="ASR model size (0.6B or 1.7B)",
    ),
    tts_model: str = typer.Option(
        "1.7B",
        "--tts-model",
        help="TTS model size (0.6B or 1.7B)",
    ),
    max_segment_duration: Optional[float] = typer.Option(
        None,
        "--max-segment-duration",
        help="Maximum segment duration in seconds (higher is faster, but less precise timing)",
    ),
    max_translation_retries: Optional[int] = typer.Option(
        None,
        "--max-translation-retries",
        help="Retries to refit TTS duration per segment (lower is faster)",
    ),
    segment_extract_workers: Optional[int] = typer.Option(
        None,
        "--segment-extract-workers",
        help="CPU workers for parallel segment extraction with FFmpeg (0 = auto)",
    ),
    keep_background: bool = typer.Option(
        False,
        "--keep-background",
        help="Keep original background audio under translated speech",
    ),
    background_volume: Optional[float] = typer.Option(
        None,
        "--background-volume",
        help="Background mix volume in range 0.0-1.0 (used with --keep-background)",
    ),
    embed_subtitles: bool = typer.Option(
        False,
        "--embed-subtitles",
        help="Burn subtitles into the output video",
    ),
    subtitle_mode: str = typer.Option(
        "translated",
        "--subtitle-mode",
        help="Subtitle mode: original, translated, both",
    ),
    notify_complete: bool = typer.Option(
        True,
        "--notify-complete/--no-notify-complete",
        help="Emit terminal bell when full translation is finished",
    ),
) -> None:
    """Full video translation pipeline."""
    config = _build_config()

    # Select models
    if asr_model == "0.6B":
        config.qwen_asr_model = "Qwen/Qwen3-ASR-0.6B"
    else:
        config.qwen_asr_model = "Qwen/Qwen3-ASR-1.7B"
    
    if tts_model == "0.6B":
        config.qwen_tts_model = "Qwen/Qwen3-TTS-25Hz-0.6B-Base"
    else:
        config.qwen_tts_model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    config.use_vad = not disable_vad
    if max_segment_duration is not None:
        config.max_segment_duration = max(1.0, float(max_segment_duration))
    if max_translation_retries is not None:
        config.max_translation_retries = max(0, int(max_translation_retries))
    if segment_extract_workers is not None:
        config.segment_extract_workers = max(0, int(segment_extract_workers))
    subtitle_mode_normalized = (subtitle_mode or "translated").strip().lower()
    if subtitle_mode_normalized not in {"original", "translated", "both"}:
        raise typer.BadParameter(
            "Invalid value for --subtitle-mode. Use: original, translated, both"
        )
    config.keep_background_audio = keep_background
    if background_volume is not None:
        config.background_audio_volume = min(max(float(background_volume), 0.0), 1.0)
    config.embed_subtitles = embed_subtitles
    config.subtitle_mode = subtitle_mode_normalized
    
    set_config(config)
    
    console.print(f"🎬 Translating video: [bold]{input_path.name}[/bold]")
    console.print(f"📍 Target language: [bold]{target_language}[/bold]")
    if source_language:
        console.print(f"🗣️ Source language (forced): [bold]{source_language}[/bold]")
    console.print(f"🧠 VAD segmentation: [bold]{'on' if config.use_vad else 'off'}[/bold]")
    console.print(f"⏱️ Max segment duration: [bold]{config.max_segment_duration:.1f}s[/bold]")
    console.print(f"🔁 Max TTS fit retries: [bold]{config.max_translation_retries}[/bold]")
    workers_label = "auto" if config.segment_extract_workers <= 0 else str(config.segment_extract_workers)
    console.print(f"🧵 Segment extract workers: [bold]{workers_label}[/bold]")
    console.print(f"🎚️ Keep background: [bold]{'on' if config.keep_background_audio else 'off'}[/bold]")
    if config.keep_background_audio:
        console.print(f"🎚️ Background volume: [bold]{config.background_audio_volume:.2f}[/bold]")
    console.print(f"📝 Embed subtitles: [bold]{'on' if config.embed_subtitles else 'off'}[/bold]")
    if config.embed_subtitles or (not no_subtitles):
        console.print(f"📝 Subtitle mode: [bold]{config.subtitle_mode}[/bold]")
    
    started_perf = time.perf_counter()

    try:
        translator = VideoTranslator(config=config)
        
        result = translator.translate_video(
            input_path=input_path,
            target_language=target_language,
            source_language=source_language,
            output_dir=output_dir,
            voice_clone=not no_voice_clone,
            generate_subtitles=not no_subtitles,
            speaker=speaker,
            keep_background=config.keep_background_audio,
            background_volume=config.background_audio_volume,
            embed_subtitles=config.embed_subtitles,
            subtitle_mode=config.subtitle_mode,
        )
        
        console.print("\n[green]✓ Translation complete![/green]")
        console.print(f"\n🎬 Translated video: {result.video_path}")
        console.print(f"🎵 Generated audio: {result.audio_path}")
        console.print(f"📝 Transcript: {result.transcript_path}")
        
        if result.subtitle_path:
            console.print(f"📹 Subtitles: {result.subtitle_path}")
        
        console.print(f"\n📊 Languages: {result.original_language} → {result.target_language}")
        elapsed_seconds = time.perf_counter() - started_perf
        finished_at = datetime.now()
        console.print(
            f"⏱️ Completed at: {finished_at.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(elapsed {elapsed_seconds:.1f}s)"
        )
        if notify_complete:
            _notify_completion()
        
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        logger.exception("Translation failed")
        raise typer.Exit(1)


@app.command("align")
def align(
    audio_path: Path = typer.Argument(
        ...,
        help="Path to audio file",
        exists=True,
    ),
    text: str = typer.Argument(
        ...,
        help="Text to align with audio",
    ),
    language: str = typer.Option(
        "English",
        "--language",
        "-l",
        help="Language of the text",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for alignment results (JSON)",
    ),
) -> None:
    """Align audio with text to get word-level timestamps."""
    config = _build_config()
    set_config(config)

    console.print(f"📍 Aligning: [bold]{audio_path.name}[/bold]")
    
    try:
        translator = VideoTranslator(config=config)
        
        result = translator.align_audio_text(
            audio_path=audio_path,
            text=text,
            language=language,
        )
        
        console.print(f"\n[green]✓ Alignment complete![/green]")
        console.print(f"📊 Segments: {len(result.segments)}")
        console.print(f"⏱️ Duration: {result.end_time - result.start_time:.2f}s")
        
        # Show first few segments
        if result.segments:
            console.print("\n[dim]First segments:[/dim]")
            for seg in result.segments[:5]:
                console.print(f"  {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")
        
        # Save to file if requested
        if output_path:
            import json
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "text": result.text,
                    "language": result.language,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "segments": result.segments,
                }, f, indent=2)
            console.print(f"💾 Saved to: {output_path}")
        
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("info")
def show_info() -> None:
    """Show system and configuration information."""
    import torch
    
    console.print("[bold]Video Translator System Info[/bold]\n")
    
    console.print(f"📦 Version: {__version__}")
    console.print(f"🐍 Python: {sys.version.split()[0]}")
    console.print(f"🔥 PyTorch: {torch.__version__}")
    console.print(f"🚀 CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    
    console.print("\n[bold]Hardware:[/bold]")
    if torch.cuda.is_available():
        console.print(f"  GPU: {torch.cuda.get_device_name(0)}")
        console.print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        console.print("  GPU: Not available")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        console.print("  Apple MPS: Available")
    
    # Check FFmpeg
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        version = result.stdout.split()[2] if result.returncode == 0 else "Unknown"
        console.print(f"🎬 FFmpeg: {version}")
    except FileNotFoundError:
        console.print("🎬 FFmpeg: Not found")


def main_entry() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main_entry()
