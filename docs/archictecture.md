# Video Translator Application - Technical Architecture

## Overview

The Video Translator is an AI-powered pipeline that translates video content from source language to target language while maintaining temporal synchronization. It uses Qwen3 models (ASR, TTS, ForcedAligner) and Silero VAD for voice activity detection.

## High-Level Architecture

```mermaid
flowchart TD
    subgraph Input
        A[Video File]
    end
    
    subgraph Audio Processing
        B[Audio Extraction]
        C[Silero VAD]
        D[Segment Extraction]
    end
    
    subgraph AI Processing
        E[Qwen ASR]
        F[Forced Alignment]
        G[NLLB Translation]
        H[Qwen TTS]
    end
    
    subgraph Output
        I[Timeline Assembly]
        J[Video Remux]
        K[Subtitles]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    F --> H
    G --> H
    H --> I
    I --> J
    I --> K
```

## Core Components

### 1. [`VideoTranslator`](src/video_translator/pipeline.py:158) - Main Pipeline Orchestrator

The central class that coordinates all translation steps:

```mermaid
classDiagram
    class VideoTranslator {
        -Config config
        -QwenASR asr
        -QwenTTS tts
        -QwenForcedAligner aligner
        -AudioProcessor audio_processor
        -VideoProcessor video_processor
        -SubtitleGenerator subtitle_generator
        -SileroVAD vad
        -SegmentQA segment_qa
        +transcribe(video_path) TranscriptionResult
        +translate_video(input_path, target_language) TranslationResult
        +synthesize_speech(text, output_path) TTSResult
    }
    
    class Config {
        +device: str
        +audio_sample_rate: int
        +use_vad: bool
        +max_segment_duration: float
        +min_segment_duration: float
    }
    
    class QwenASR {
        +transcribe(audio, sample_rate) ASRResult
    }
    
    class QwenTTS {
        +synthesize(text, language) TTSResult
        +synthesize_voice_clone(text, reference_audio) TTSResult
        +synthesize_voice_design(text, voice_description) TTSResult
    }
    
    class SileroVAD {
        +detect(audio_path) List[SpeechRegion]
    }
    
    VideoTranslator --> Config
    VideoTranslator --> QwenASR
    VideoTranslator --> QwenTTS
    VideoTranslator --> SileroVAD
```

### 2. [`translate_video()`](src/video_translator/pipeline.py:849) - Main Translation Pipeline

The complete translation workflow:

```mermaid
flowchart TD
    Start[Start Translation] --> Step1[Step 1: Extract Audio]
    Step1 --> Step2[Step 2: VAD Detection]
    Step2 --> Step3[Step 3: Build Processing Regions]
    Step3 --> Step4[Step 4: Segment Processing Loop]
    
    Step4 --> ASR[ASR per Segment]
    ASR --> Translate[Translate Text]
    Translate --> Fit[_fit_translation_to_duration]
    Fit --> TTS[TTS Synthesis]
    TTS --> Retime[Retiming Check]
    
    Retime -->|Fits| Timeline[Timeline Assembly]
    Retime -->|Needs adjustment| Fit
    Retime -->|Large error| Fit
    
    Timeline --> Step5[Step 5: Final Muxing]
    Step5 --> Step6[Step 6: Generate Subtitles & QA]
    Step6 --> End[Translation Complete]
```

### 3. VAD-Based Segmentation

Voice Activity Detection creates speech regions for processing:

```mermaid
flowchart TD
    Audio[Audio File] --> VAD[Silero VAD]
    VAD -->|Speech Regions| Merge[Merge Adjacent Regions]
    Merge -->|Long regions| Split[Split by max_duration]
    Split --> Filter[Filter by min_duration]
    Filter --> Regions[Processing Regions]
    
    VAD -->|Fallback| Fixed[Fixed Timeline Chunks]
    Filter -->|No valid regions| Fixed
```

### 4. Segment Processing Flow

Each speech segment goes through this pipeline:

```mermaid
flowchart TD
    Segment[Segment] --> Extract[Extract Audio]
    Extract --> ASR[Qwen ASR Transcribe]
    ASR -->|Text + Timestamps| Align[Forced Alignment]
    Align -->|Refined Timestamps| Translate[NLLB Translation]
    Translate --> Fit[_fit_translation_to_duration]
    Fit -->|Compact| TTS[Qwen TTS]
    TTS --> Measure[Measure Duration]
    Measure -->|Within tolerance| Accept[Accept]
    Measure -->|Slight error| Retime[Time-Stretch]
    Retime --> Accept
    Measure -->|Large error| Fit
    Accept --> Save[Save Segment Audio]
```

### 5. Duration Control Loop

Critical for maintaining synchronization:

```mermaid
flowchart TD
    Source[Source Text + Target Window] --> Translate[Translate]
    Translate --> Estimate[Estimate Duration]
    Estimate --> Check1{Fits Window?}
    
    Check1 -->|Yes| TTS[TTS Synthesis]
    Check1 -->|No| Rewrite[Rewrite Compact]
    Rewrite --> Estimate
    
    TTS --> Measure[Measure Actual Duration]
    Measure --> Check2{Error Tolerance?}
    
    Check2 -->|Within| Accept[Accept Segment]
    Check2 -->|Small| Retime[Apply Mild Retiming]
    Retime --> Accept
    Check2 -->|Large| Rewrite
```

### 6. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input
        Video[Video.mp4]
    end
    
    subgraph Intermediate
        Audio[Audio.wav]
        Segments[Segment_00001.wav...]
    end
    
    subgraph AI Outputs
        Transcript[Transcript.txt]
        Translated[Translated.txt]
        Subtitles[Subtitles.srt]
    end
    
    subgraph Output
        Final[Video_translated.mp4]
    end
    
    Video --> Audio
    Audio --> Segments
    Segments -->|ASR| Transcript
    Transcript -->|Translation| Translated
    Segments -->|TTS| Final
    Translated -->|SRT| Subtitles
    Subtitles --> Final
```

## Key Models

### [`QwenASR`](src/video_translator/models/asr.py:34) - Speech Recognition

- Uses `Qwen/Qwen3-ASR-1.7B` model
- Supports 52 languages
- Provides word-level timestamps
- Optional forced aligner integration

### [`QwenTTS`](src/video_translator/models/tts.py:31) - Text-to-Speech

- Uses `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Three modes:
  - **Basic**: Preset voices
  - **Voice Clone**: From reference audio
  - **Voice Design**: From natural language description

### [`SileroVAD`](src/video_translator/processing/vad.py:29) - Voice Activity Detection

- Uses Silero VAD model
- Falls back to energy-based detection if unavailable
- Configurable thresholds for speech detection

### [`SegmentQA`](src/video_translator/processing/qa.py:23) - Quality Assurance

Validates segment timing and audio quality:
- Duration mismatch detection
- Gap/overlap detection
- Clipping detection

## Configuration

The [`Config`](src/video_translator/config.py:11) class manages settings via Pydantic:

```python
Config(
    device="auto",              # cuda/mps/cpu
    audio_sample_rate=16000,    # Hz
    use_vad=True,               # Enable VAD
    max_segment_duration=30.0,  # seconds
    min_segment_duration=1.0,   # seconds
    vad_threshold=0.5,          # VAD confidence threshold
)
```

## CLI Usage

```bash
# Transcribe video
video-translator transcribe input.mp4

# Translate video
video-translator translate input.mp4 --target-language spanish

# TTS from text
video-translator tts input.txt --language Spanish
```

## File Structure

```
src/video_translator/
├── pipeline.py          # Main VideoTranslator class
├── config.py            # Configuration management
├── cli.py               # CLI entry point
├── models/
│   ├── asr.py          # QwenASR wrapper
│   ├── tts.py          # QwenTTS wrapper
│   └── aligner.py      # QwenForcedAligner wrapper
└── processing/
    ├── vad.py          # SileroVAD implementation
    ├── audio.py        # FFmpeg audio processing
    ├── video.py        # FFmpeg video processing
    ├── subtitles.py    # SRT/VTT generation
    └── qa.py           # Quality assurance checks
```

## Technical Highlights

1. **Modular Architecture**: Each component is independently testable
2. **VAD Fallback**: Energy-based VAD if Silero unavailable
3. **Translation Caching**: NLLB model loaded once per pipeline
4. **Device Auto-Detection**: Automatically selects cuda/mps/cpu
5. **Segment Parallelization**: Audio extraction uses ThreadPoolExecutor
6. **Duration Control**: Iterative retry loop for timing fit
7. **QA Integration**: Automatic validation of all segments

This architecture enables processing of long-form videos (2-3 hours) by breaking them into manageable segments while maintaining temporal synchronization with the original video.