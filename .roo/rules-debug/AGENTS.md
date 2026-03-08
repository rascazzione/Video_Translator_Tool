# AGENTS.md - Debug Mode

This file provides debugging-specific guidance for agents working in this repository.

## Non-Obvious Debugging Rules

1. **Silent VAD fallback**: If [`SileroVAD`](src/video_translator/processing/vad.py) produces unusable output, the pipeline silently falls back to fixed timeline chunks. Check VAD output first when timing seems wrong.

2. **Translation model caching**: NLLB model loads once in [`SegmentQA`](src/video_translator/processing/qa.py). If translation fails, check if model initialized correctly - it reuses the same instance.

3. **Device selection**: Device defaults to `"auto"` which selects cuda/mps/cpu. Check `DEVICE` env var or `--device` CLI option when GPU isn't used.

4. **Config case-insensitivity**: Config keys in `.env` are case-insensitive but environment variables are uppercase (`QWEN_ASR_MODEL`, not `qwen_asr_model`).

5. **TTS mode conflicts**: Don't use `voice_clone` and `voice_design` together - they are mutually exclusive modes.

6. **Missing FFmpeg**: Pipeline requires FFmpeg in PATH. Check `ffmpeg -version` if audio extraction fails.

7. **Model download on first use**: Models download automatically on first run. Check network access and `MODEL_CACHE_DIR` if models fail to load.