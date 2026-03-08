# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build/Test Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run single test file
pytest tests/test_config.py

# Run single test
pytest tests/test_config.py::test_default_config

# Lint with ruff
ruff check .

# Format with black
black .

# Type check
mypy src/
```

## Code Style

- **Line length**: 100 characters
- **Formatter**: Black (pyproject.toml [tool.black])
- **Linter**: Ruff (E, F, W, I, N, UP, B, C4, SIM; ignores E501)
- **Type checker**: MyPy strict mode (disallow_untyped_defs enabled)
- **Config**: Pydantic v2 with pydantic-settings

## Non-Obvious Patterns

1. **Dual language mapping**: Pipeline uses two separate language maps - [`LANGUAGE_MAP`](src/video_translator/pipeline.py:24) for TTS (ISO 639-1 to full name) and [`NLLB_LANGUAGE_MAP`](src/video_translator/pipeline.py:38) for translation (BCP-47 codes). Don't confuse them.

2. **VAD fallback**: [`SileroVAD`](src/video_translator/processing/vad.py) has a fallback to fixed timeline chunks if VAD output is unusable. This is automatic in [`VideoTranslator`](src/video_translator/pipeline.py).

3. **Translation caching**: NLLB model and tokenizer are loaded once and reused across segments (not per-segment). See [`SegmentQA`](src/video_translator/processing/qa.py).

4. **TTS modes**: Three distinct TTS modes - basic, voice cloning (`voice_clone=True` with `reference_audio`), and voice design (`voice_design=True` with `voice_description`).

5. **CLI entry points**: Both `python -m video_translator.cli` and `video-translator` (installed script) work identically.

6. **Config loading**: Config supports `.env` file (case-insensitive keys) AND environment variables. Uses [`pydantic_settings.BaseSettings`](src/video_translator/config.py:11).

7. **Device auto-detection**: Device defaults to `"auto"` which selects cuda/mps/cpu based on availability. Override with `DEVICE` env var or `--device` CLI option.