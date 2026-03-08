# AGENTS.md - Ask Mode

This file provides guidance for understanding this codebase when answering questions.

## Non-Obvious Documentation Context

1. **Dual language systems**: The pipeline uses two separate language mappings - [`LANGUAGE_MAP`](src/video_translator/pipeline.py:24) for TTS (ISO 639-1 codes like "es" → "Spanish") and [`NLLB_LANGUAGE_MAP`](src/video_translator/pipeline.py:38) for translation (BCP-47 codes like "es" → "spa_Latn").

2. **VAD segmentation**: [`SileroVAD`](src/video_translator/processing/vad.py) detects speech regions, but has automatic fallback to fixed timeline chunks if VAD output is unusable.

3. **Three TTS modes**: Basic TTS, voice cloning (`voice_clone=True` with `reference_audio`), and voice design (`voice_design=True` with `voice_description`).

4. **Translation caching**: NLLB model and tokenizer are loaded once in [`SegmentQA`](src/video_translator/processing/qa.py) and reused across all segments for efficiency.

5. **Config system**: Uses Pydantic v2 with pydantic-settings. Supports both `.env` file (case-insensitive keys) and environment variables (uppercase).

6. **Device auto-detection**: Device defaults to `"auto"` which automatically selects cuda/mps/cpu based on hardware availability.

7. **CLI entry points**: Both `python -m video_translator.cli` and the installed `video-translator` script work identically.