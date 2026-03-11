# Video Translator Architecture

## Purpose

This repository is a CLI-first video dubbing pipeline. It takes an input video, extracts speech, translates the spoken content, synthesizes new speech in the target language, and remuxes the result back into a final video.

The main runtime entrypoint is [`translate-video`](../src/video_translator/cli.py), which delegates to [`VideoTranslator.translate_video()`](../src/video_translator/pipeline.py).

## System Diagram

```mermaid
flowchart LR
    A[CLI / Python API] --> B[Config]
    B --> C[VideoTranslator]
    C --> D[AudioProcessor]
    C --> E[SileroVAD]
    C --> F[QwenASR]
    C --> G[NLLB Translation]
    C --> H[QwenTTS]
    C --> I[QwenForcedAligner]
    C --> J[SubtitleGenerator]
    C --> K[SegmentQA]
    C --> L[VideoProcessor]

    D --> M[Source audio wav]
    E --> N[Speech regions]
    F --> O[Source transcript]
    G --> P[Translated text]
    H --> Q[Synthesized segment audio]
    J --> R[SRT subtitles]
    K --> S[QA JSON report]
    L --> T[Final translated MP4]
```

## Main Runtime Flow

```mermaid
flowchart TD
    Start[translate-video] --> Extract[1. Extract source audio]
    Extract --> VAD[2. Detect speech regions with VAD]
    VAD --> Fallback{usable regions?}
    Fallback -- no --> Fixed[Build fixed timeline chunks]
    Fallback -- yes --> Regions[Merge and split VAD regions]
    Fixed --> Segments
    Regions --> Segments[3. Extract segment wav files]
    Segments --> ASR[4. Run ASR per segment]
    ASR --> Align[5. Optional forced alignment]
    Align --> Translate[6. Translate segment text]
    Translate --> Fit[7. Synthesize and fit timing]
    Fit --> Timeline[8. Assemble full translated audio timeline]
    Timeline --> BG{keep background audio?}
    BG -- yes --> Mix[Mix translated speech with original background]
    BG -- no --> Assets
    Mix --> Assets[9. Write transcript, subtitles, QA report]
    Assets --> Mux[10. Replace video audio / burn subtitles]
    Mux --> End[Final output files]
```

## Component Responsibilities

### CLI

- File: [`src/video_translator/cli.py`](../src/video_translator/cli.py)
- Provides commands for transcription, TTS, alignment, and full translation.
- Builds config from environment or a user-supplied `--config` env file.

### Config

- File: [`src/video_translator/config.py`](../src/video_translator/config.py)
- Uses `pydantic-settings`.
- Supports environment variables and `.env` files.
- Holds model, hardware, segmentation, subtitle, and output settings.

### Pipeline Orchestrator

- File: [`src/video_translator/pipeline.py`](../src/video_translator/pipeline.py)
- Lazily loads models and processing helpers.
- Coordinates extraction, segmentation, ASR, translation, TTS, subtitle generation, QA, and muxing.

### Audio and Video Processing

- Files:
  - [`src/video_translator/processing/audio.py`](../src/video_translator/processing/audio.py)
  - [`src/video_translator/processing/video.py`](../src/video_translator/processing/video.py)
- FFmpeg-backed utilities for extraction, segment slicing, time-stretching, timeline assembly, background mixing, and final remux.

### ASR / TTS / Alignment

- Files:
  - [`src/video_translator/models/asr.py`](../src/video_translator/models/asr.py)
  - [`src/video_translator/models/tts.py`](../src/video_translator/models/tts.py)
  - [`src/video_translator/models/aligner.py`](../src/video_translator/models/aligner.py)
- `QwenASR` produces transcript text and optional timestamps.
- `QwenTTS` supports preset voice, voice cloning, and voice design.
- `QwenForcedAligner` is best-effort in the current pipeline and does not yet drive downstream cue timing.

### Segmentation

- File: [`src/video_translator/processing/vad.py`](../src/video_translator/processing/vad.py)
- Primary mode uses Silero VAD.
- Fallback mode uses simple energy-based speech detection.
- If detected regions are unusable, the pipeline falls back again to fixed timeline chunks.

## Segment Processing Flow

Each extracted speech region follows this path:

```mermaid
flowchart TD
    A[segment.wav] --> B[ASR]
    B --> C[detected text and language]
    C --> D[forced aligner, best-effort]
    C --> E[translation]
    E --> F[TTS]
    F --> G{duration close enough?}
    G -- yes --> H[accept]
    G -- mild drift --> I[time-stretch audio]
    I --> H
    G -- too much drift --> J[compact text and retry]
    J --> F
```

## Outputs

The full translation pipeline emits:

- translated `.wav` audio track
- translated `.mp4` video
- translated text transcript
- `.srt` subtitles when enabled
- segment JSON report with timing and QA issues

## Current Strengths

- Clean orchestration in one pipeline class
- Good fallback behavior around segmentation
- Model loading is lazy, which keeps startup simpler
- NLLB translation backend is cached and reused within a run
- Timing fit loop is pragmatic and easy to reason about

## Current Limitations

- Forced alignment is called but not used to refine subtitles or retiming decisions
- ASR, translation, and TTS are still processed serially per segment
- Translation fallback previously became silent original-text passthrough
- Subtitle shaping helpers existed but were not applied in the main translation flow
- README/config mention API-oriented pieces that are not implemented in this repo

## Improvements Implemented In This Iteration

- `--config` now actually loads a user-provided env file in the CLI
- subtitle shaping is now applied for single-language subtitle modes:
  - merge nearby cues
  - split oversized cues
- segment reports now include translation metadata:
  - source language
  - translation status
  - translation error when the pipeline fell back to the original text

## Recommended Next Improvements

1. Make forced alignment feed real downstream timing.
2. Add resumable per-segment caching for ASR, translation, and TTS.
3. Introduce bounded concurrency for model stages.
4. Promote QA from report-only to automatic retry rules.
5. Add explicit translation failure policy: warn, retry, or hard-fail.
6. Either implement the API/worker stack or remove those claims from docs and config.
