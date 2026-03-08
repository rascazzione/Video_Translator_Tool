# AGENTS.md - Code Mode

This file provides coding-specific guidance for agents working in this repository.

## Non-Obvious Coding Rules

1. **Language mapping**: Use [`LANGUAGE_MAP`](src/video_translator/pipeline.py:24) for TTS (ISO 639-1 to full name) and [`NLLB_LANGUAGE_MAP`](src/video_translator/pipeline.py:38) for translation (BCP-47 codes). These are different mappings.

2. **TTS modes**: Three distinct modes exist - basic, voice cloning (`voice_clone=True` with `reference_audio`), and voice design (`voice_design=True` with `voice_description`). Don't mix parameters.

3. **Config loading**: Uses [`pydantic_settings.BaseSettings`](src/video_translator/config.py:11) with case-insensitive `.env` keys. Both `.env` file and environment variables work.

4. **Device auto-detection**: Device defaults to `"auto"` which selects cuda/mps/cpu. Override via `DEVICE` env var or `--device` CLI option.

5. **Translation caching**: NLLB model/tokenizer loaded once in [`SegmentQA`](src/video_translator/processing/qa.py) and reused across segments - don't reload per-segment.

6. **VAD fallback**: [`SileroVAD`](src/video_translator/processing/vad.py) automatically falls back to fixed timeline chunks if VAD output is unusable. This is handled in [`VideoTranslator`](src/video_translator/pipeline.py).