"""Command-line interface for Video Translator."""

import logging
import sys
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

# Create Typer app
app = typer.Typer(
    name="video-translator",
    help="Video translation pipeline using Qwen3 models",
    add_completion=False,
)


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
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if config_file and config_file.exists():
        # Load config from file
        # In production, parse and set config
        logger.info(f"Config file: {config_file}")


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
    config = Config()
    
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
    config = Config()
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
) -> None:
    """Full video translation pipeline."""
    config = Config()
    
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
    
    set_config(config)
    
    console.print(f"🎬 Translating video: [bold]{input_path.name}[/bold]")
    console.print(f"📍 Target language: [bold]{target_language}[/bold]")
    console.print(f"🧠 VAD segmentation: [bold]{'on' if config.use_vad else 'off'}[/bold]")
    
    try:
        translator = VideoTranslator(config=config)
        
        result = translator.translate_video(
            input_path=input_path,
            target_language=target_language,
            output_dir=output_dir,
            voice_clone=not no_voice_clone,
            generate_subtitles=not no_subtitles,
            speaker=speaker,
        )
        
        console.print("\n[green]✓ Translation complete![/green]")
        console.print(f"\n🎬 Translated video: {result.video_path}")
        console.print(f"🎵 Generated audio: {result.audio_path}")
        console.print(f"📝 Transcript: {result.transcript_path}")
        
        if result.subtitle_path:
            console.print(f"📹 Subtitles: {result.subtitle_path}")
        
        console.print(f"\n📊 Languages: {result.original_language} → {result.target_language}")
        
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
    config = Config()
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
