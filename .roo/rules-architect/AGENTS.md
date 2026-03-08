# AGENTS.md - Architect Mode

This file provides architectural guidance for agents working in this repository.

## Non-Obvious Architectural Constraints

1. **Dual language mapping architecture**: Two separate language maps exist - [`LANGUAGE_MAP`](src/video_translator/pipeline.py:24) for TTS (ISO 639-1 to full name) and [`NLLB_LANGUAGE_MAP`](src/video_translator/pipeline.py:38) for translation (BCP-47 codes). This is a critical design distinction.

2. **VAD with automatic fallback**: [`SileroVAD`](src/video_translator/processing/vad.py) provides speech segmentation but has built-in fallback to fixed timeline chunks. This is handled automatically in [`VideoTranslator`](src/video_translator/pipeline.py) - no manual intervention needed.

3. **Translation model singleton**: NLLB model and tokenizer are loaded once in [`SegmentQA`](src/video_translator/processing/qa.py) and reused across all segments. This is an optimization, not a bug - don't refactor to load per-segment.

4. **Three TTS modes are mutually exclusive**: Basic TTS, voice cloning (`voice_clone=True` with `reference_audio`), and voice design (`voice_design=True` with `voice_description`) cannot be combined.

5. **Config uses Pydantic v2**: Configuration is handled by [`pydantic_settings.BaseSettings`](src/video_translator/config.py:11) with support for both `.env` files (case-insensitive) and environment variables (uppercase).

6. **Device auto-detection**: The `"auto"` device option automatically selects cuda/mps/cpu based on hardware availability - this is the recommended default.

7. **Pipeline flow**: Video → Audio extraction → VAD segmentation → ASR → Translation (NLLB) → TTS → QA → Video muxing. Each stage is modular and can be used independently.