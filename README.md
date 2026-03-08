# 🎬 Video Translator

An open-source video translation pipeline using **Qwen3** family models (ASR, TTS, ForcedAligner) for complete audio transcription, translation, and speech synthesis.

## Features

- 🎙️ **Speech-to-Text** - Qwen3-ASR with 52 language support
- 🔍 **VAD Segmentation** - Silero VAD for speech-bounded long-form processing
- 🛟 **Robust Segmentation Fallback** - if VAD output is unusable, the pipeline falls back to fixed timeline chunks
- 📝 **Forced Alignment** - Word-level timestamps with Qwen3-ForcedAligner
- 🗣️ **Text-to-Speech** - Qwen3-TTS with voice cloning and design
- ⏱️ **Duration Control** - Segment-level timing fit with mild retiming
- ⚡ **Cached Translation Backend** - NLLB model/tokenizer are loaded once and reused across segments
- 🔄 **Full Pipeline** - Video → VAD → ASR → Translation → TTS → QA → Output
- 📹 **FFmpeg Integration** - Audio extraction and video muxing
- 💻 **CLI & API** - Command-line and REST API interfaces

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg installed and in PATH
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ VRAM for 0.6B models, 16GB+ for 1.7B models

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video_translator.git
cd video_translator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download models (optional - models download on first use)
python scripts/download_models.py
```

### Basic Usage

```bash
# You can also use `video-translator` instead of `python -m video_translator.cli`

# 1) Transcribe audio from a video (outputs transcript + optional .srt)
python -m video_translator.cli transcribe video.mp4 --output ./output --language en

# 2) Generate speech from a text file
python -m video_translator.cli tts text.txt --output ./output/speech.wav --language English --speaker Aiden

# 3) Full video translation pipeline (VAD-enabled segmented dubbing)
python -m video_translator.cli translate-video \
    video.mp4 \
    es \
    --output ./output

# 4) Optional: disable VAD and use fixed timeline chunks
python -m video_translator.cli translate-video \
    video.mp4 \
    es \
    --output ./output \
    --disable-vad

# 5) Speed-focused run (fewer, larger segments + no TTS fit retries)
python -m video_translator.cli translate-video \
    video.mp4 \
    es \
    --output ./output \
    --max-segment-duration 45 \
    --max-translation-retries 0 \
    --segment-extract-workers 6

# 6) Maximum speed preset (lower quality/fidelity)
python -m video_translator.cli translate-video \
    video.mp4 \
    es \
    --output ./output \
    --max-segment-duration 60 \
    --max-translation-retries 0 \
    --no-voice-clone \
    --asr-model 0.6B \
    --tts-model 0.6B
```

## Project Structure

```
video_translator/
├── src/video_translator/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── pipeline.py         # Main pipeline orchestrator
│   ├── models/
│   │   ├── asr.py          # Qwen3-ASR wrapper
│   │   ├── tts.py          # Qwen3-TTS wrapper
│   │   └── aligner.py      # Qwen3-ForcedAligner wrapper
│   ├── processing/
│   │   ├── audio.py        # Audio extraction
│   │   ├── video.py        # Video muxing
│   │   └── subtitles.py    # SRT generation
│   └── cli.py              # CLI interface
├── scripts/
│   └── download_models.py  # Model download script
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Configuration

Create a `.env` file or set environment variables:

```bash
# Model Configuration
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B

# Hardware Configuration
DEVICE=cuda              # cuda, mps, cpu
PRECISION=bf16           # bf16, fp16, fp32
FLASH_ATTENTION=true

# Storage
MODEL_CACHE_DIR=./models_cache
OUTPUT_DIR=./output

# Segmentation/Timing Controls
MAX_SEGMENT_DURATION=30.0
MIN_SEGMENT_DURATION=0.4
MAX_TRANSLATION_RETRIES=2
```

## CLI Speed Tuning

Useful `translate-video` options for performance tuning:

- `--max-segment-duration FLOAT`: larger chunks reduce per-segment overhead (faster, less precise timing)
- `--max-translation-retries INT`: retry count for duration fitting (set `0` for speed)
- `--segment-extract-workers INT`: parallel CPU workers for FFmpeg segment extraction
- `--disable-vad`: skip speech-only detection and chunk the full timeline
- `--no-voice-clone`: much faster than per-segment voice cloning
- `--asr-model 0.6B` / `--tts-model 0.6B`: smaller models, lower latency

## API Reference

### Transcription

```python
from video_translator import VideoTranslator

translator = VideoTranslator()
result = translator.transcribe("video.mp4")
print(result.text)
print(result.timestamps)  # Word-level timestamps
```

### Text-to-Speech

```python
from video_translator import VideoTranslator

translator = VideoTranslator()

# Basic TTS
audio = translator.tts("Hello, world!", language="English")

# Voice cloning
audio = translator.tts(
    "Hello, I'm cloned!",
    voice_clone=True,
    reference_audio="reference.wav"
)

# Voice design
audio = translator.tts(
    "I'm a pirate!",
    voice_design=True,
    voice_description="gruff pirate voice"
)
```

### Full Pipeline

```python
from video_translator import VideoTranslator

translator = VideoTranslator()

result = translator.translate_video(
    input_path="english_video.mp4",
    target_language="spanish",
    voice_clone=True,  # Clone original speaker's voice
    generate_subtitles=True
)

print(f"Output: {result.video_path}")
print(f"Subtitles: {result.subtitle_path}")
```

## Model Specifications

| Model | Parameters | VRAM | Languages |
|-------|------------|------|-----------|
| Qwen3-ASR-0.6B | 600M | 4-6GB | 52 |
| Qwen3-ASR-1.7B | 1.7B | 6-8GB | 52 |
| Qwen3-TTS-0.6B | 600M | 4-6GB | 10 |
| Qwen3-TTS-1.7B | 1.7B | 6-8GB | 10 |
| Qwen3-ForcedAligner-0.6B | 600M | ~2GB | 11 |

## Performance

Expected processing times (per minute of video, RTX 3090):

| Component | Time |
|-----------|------|
| ASR Transcription | 10-30s |
| TTS Generation | 15-40s |
| Full Pipeline | 30-80s |

## License

Apache 2.0

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Architecture Documentation](plans/video_translator_architecture.md)
