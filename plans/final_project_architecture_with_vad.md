# Long-Form Video Audio Translation and Dubbing System
## Final Technical Overview for a Developer Reader

## 1. Project goal

Build an application that takes a long video, such as a 2–3 hour interview, podcast, lecture, documentary, or film-like asset, and produces a translated audio track that stays synchronized with the original video timeline.

The core requirement is not only translation accuracy. The system must also preserve temporal alignment with the image, which means:

- each spoken intervention should start and end in the correct time window,
- pauses should feel natural,
- the generated translated voice should fit the original scene timing,
- the final dubbed track should be mixable back into the original video.

This is a long-form batch processing system, not a simple real-time subtitle translator.

---

## 2. Design principle

For long videos, the correct architecture is modular.

Do not process the entire video as one monolithic audio job.  
Instead, process the video as a sequence of time-bounded speech segments.

The main pipeline is:

1. ingest video,
2. extract and preprocess audio,
3. detect where speech exists with an open-source VAD,
4. split and align speaker utterances,
5. run ASR,
6. refine timestamps,
7. translate with duration constraints,
8. synthesize speech,
9. retime when necessary,
10. mix and export.

The VAD is important. In this design, use an **open-source, current VAD**, specifically **Silero VAD**, as the speech activity detector.

---

## 3. Why VAD is necessary

ASR and alignment are not enough by themselves.

Qwen ASR can transcribe and produce timestamps.  
A forced aligner can refine those timestamps.  
Qwen TTS can synthesize translated speech.

But none of those components is the right tool for deciding where speech actually starts and stops in a noisy long-form track.

A dedicated VAD provides:

- cleaner initial segmentation,
- better preservation of silence and pacing,
- lower ASR cost by skipping non-speech regions,
- fewer bad boundaries,
- better system stability over thousands of segments.

For this reason, VAD should be part of the production architecture from the start.

---

## 4. Recommended stack

### Core models and services

- **Silero VAD** for voice activity detection
- **Speaker diarization** for speaker segmentation
- **Qwen ASR** for transcription and coarse timestamps
- **Forced alignment** for timestamp refinement
- **Translation service** with duration-aware rewriting
- **Qwen TTS** for target-language speech synthesis
- **Time-stretch / retiming** for final fit correction
- **Mixing and remuxing** for final delivery

### Infrastructure

- **FastAPI** or similar backend API
- **Job orchestrator**
- **Queue / broker** such as Redis-backed workers
- **PostgreSQL** for metadata and state
- **Object storage** for intermediate and final assets
- **FFmpeg** for extraction, remux, audio handling

---

## 5. High-level component diagram

```mermaid
flowchart LR

    UI[Frontend / Review Editor]
    API[API Gateway]
    AUTH[Auth]
    ORCH[Job Orchestrator]
    QUEUE[Queue / Broker]
    DB[(PostgreSQL)]
    OBJ[(Object Storage)]

    INGEST[Ingest Service]
    FFMPEG[FFmpeg Audio Extraction]
    SEP[Vocal Separation / Stem Prep]
    VAD[Silero VAD]
    DIAR[Diarization]

    ASR[Qwen ASR Service]
    ALIGN[Forced Alignment Service]
    TRANS[Translation Service]
    DUR[Duration Control / Compression]
    TTS[Qwen TTS Service]
    RETIME[Retiming / Time-Stretch]
    MIX[Mixing / Mastering]
    MUX[Video Remux / Export]

    QA[QA / Validation]
    OBS[Monitoring / Logs / Metrics]

    UI --> API
    API --> AUTH
    API --> ORCH
    API --> DB
    API --> OBJ

    ORCH --> QUEUE
    ORCH --> DB

    QUEUE --> INGEST
    INGEST --> FFMPEG
    FFMPEG --> SEP
    SEP --> VAD
    VAD --> DIAR
    DIAR --> ASR
    ASR --> ALIGN
    ALIGN --> TRANS
    TRANS --> DUR
    DUR --> TTS
    TTS --> RETIME
    RETIME --> MIX
    MIX --> MUX

    ASR --> QA
    ALIGN --> QA
    TRANS --> QA
    TTS --> QA
    RETIME --> QA
    MIX --> QA

    QA --> DB
    MUX --> OBJ
    MUX --> DB

    API --> OBS
    ORCH --> OBS
    ASR --> OBS
    TTS --> OBS
    QA --> OBS
```

---

## 6. End-to-end pipeline flow

```mermaid
flowchart TD

    A[Upload video] --> B[Extract audio and metadata]
    B --> C[Optional vocal separation]
    C --> D[Run Silero VAD]
    D --> E[Build speech regions]
    E --> F[Run diarization]
    F --> G[Create segment candidates]
    G --> H[Run Qwen ASR]
    H --> I[Forced alignment refinement]
    I --> J[Reconstruct utterances]

    J --> K[Translate]
    K --> L[Apply duration constraints]
    L --> M{Fits target time window?}

    M -- Yes --> N[Run Qwen TTS]
    M -- No --> O[Rewrite compact translation]
    O --> L

    N --> P[Measure actual audio duration]
    P --> Q{Timing error acceptable?}

    Q -- Yes --> R[Place on timeline]
    Q -- Slight error --> S[Apply mild retiming]
    S --> R
    Q -- Large error --> O

    R --> T[Mix with music and effects]
    T --> U[Remux with original video]
    U --> V[Run QA]
    V --> W{Passed?}

    W -- Yes --> X[Export final asset]
    W -- No --> Y[Flag problematic segments]
    Y --> Z[Manual correction or selective regeneration]
    Z --> N
```

---

## 7. Job state diagram

```mermaid
stateDiagram-v2
    [*] --> Uploaded
    Uploaded --> Validating
    Validating --> Queued
    Queued --> Preprocessing
    Preprocessing --> VADRunning
    VADRunning --> DiarizationRunning
    DiarizationRunning --> ASRRunning
    ASRRunning --> Aligning
    Aligning --> Translating
    Translating --> Synthesizing
    Synthesizing --> Mixing
    Mixing --> QAReview
    QAReview --> Exporting
    Exporting --> Completed

    Validating --> Failed
    Preprocessing --> Failed
    VADRunning --> Failed
    DiarizationRunning --> Failed
    ASRRunning --> Failed
    Aligning --> Failed
    Translating --> Failed
    Synthesizing --> Failed
    Mixing --> Failed
    QAReview --> NeedsManualReview
    NeedsManualReview --> Translating
    NeedsManualReview --> Synthesizing
    NeedsManualReview --> Mixing
    Failed --> Retrying
    Retrying --> Queued
    Failed --> Cancelled
    Completed --> [*]
    Cancelled --> [*]
```

---

## 8. Segment state diagram

In production, the real unit of work is the segment, not the whole video.

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> VADBounded
    VADBounded --> SpeakerAssigned
    SpeakerAssigned --> ASRDone
    ASRDone --> Aligned
    Aligned --> Translated
    Translated --> DurationChecked

    DurationChecked --> TTSPending: fits
    DurationChecked --> RewriteNeeded: does_not_fit

    RewriteNeeded --> Translated

    TTSPending --> TTSDone
    TTSDone --> DurationMeasured

    DurationMeasured --> Accepted: low_error
    DurationMeasured --> StretchNeeded: medium_error
    DurationMeasured --> RewriteNeeded: high_error

    StretchNeeded --> Stretched
    Stretched --> Accepted

    Accepted --> Mixed
    Mixed --> QAOk
    Mixed --> QAFlagged

    QAFlagged --> RewriteNeeded
    QAOk --> ExportReady
    ExportReady --> [*]
```

---

## 9. Sequence diagram

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as API
    participant ORCH as Orchestrator
    participant Q as Queue
    participant PRE as Preprocess Worker
    participant VAD as Silero VAD Worker
    participant DIAR as Diarization Worker
    participant ASR as Qwen ASR Worker
    participant AL as Alignment Worker
    participant TR as Translation Worker
    participant DC as Duration Control
    participant TTS as Qwen TTS Worker
    participant MX as Mixing Worker
    participant DB as DB / Object Storage

    U->>FE: Upload video
    FE->>API: createJob(video)
    API->>DB: store metadata
    API->>ORCH: start job
    ORCH->>Q: enqueue preprocess

    Q->>PRE: extract audio and prepare stems
    PRE->>DB: save audio assets

    ORCH->>Q: enqueue VAD
    Q->>VAD: detect speech regions
    VAD->>DB: save speech intervals

    ORCH->>Q: enqueue diarization
    Q->>DIAR: assign speakers
    DIAR->>DB: save speaker intervals

    loop per segment
        ORCH->>Q: enqueue ASR
        Q->>ASR: transcribe(segment)
        ASR->>DB: save transcript and coarse timestamps

        ORCH->>Q: enqueue alignment
        Q->>AL: refine timestamps
        AL->>DB: save aligned words / utterances

        ORCH->>Q: enqueue translation
        Q->>TR: translate(text, target_duration)
        TR->>DB: save translation

        ORCH->>Q: enqueue duration control
        Q->>DC: validate target fit
        DC->>DB: fit result or rewrite needed

        alt fits
            ORCH->>Q: enqueue TTS
            Q->>TTS: synthesize(translated_text)
            TTS->>DB: save generated waveform
        else does not fit
            DC->>TR: request compact rewrite
        end
    end

    ORCH->>Q: enqueue mixing
    Q->>MX: assemble timeline and mix
    MX->>DB: save final output

    API->>FE: progress and output
    FE->>U: preview and download
```

---

## 10. Deployment diagram

```mermaid
flowchart TB

    subgraph Client
        FE[Web App / Review UI]
    end

    subgraph Backend
        API[FastAPI / API Gateway]
        ORCH[Orchestrator]
        REDIS[Redis / Broker]
        PG[(PostgreSQL)]
        S3[(S3 / MinIO)]
    end

    subgraph CPU_Workers
        PREW[Preprocess Worker]
        VADW[Silero VAD Worker]
        DIARW[Diarization Worker]
        ALIGNW[Alignment Worker]
        MIXW[Mix Worker]
        QAW[QA Worker]
    end

    subgraph GPU_Workers
        ASRW[Qwen ASR Worker]
        TRW[Translation Worker]
        TTSW[Qwen TTS Worker]
    end

    subgraph Observability
        MON[Metrics / Logs / Traces]
    end

    FE --> API
    API --> ORCH
    API --> PG
    API --> S3

    ORCH --> REDIS
    ORCH --> PG

    REDIS --> PREW
    REDIS --> VADW
    REDIS --> DIARW
    REDIS --> ALIGNW
    REDIS --> MIXW
    REDIS --> QAW
    REDIS --> ASRW
    REDIS --> TRW
    REDIS --> TTSW

    PREW --> S3
    PREW --> PG

    VADW --> PG
    DIARW --> PG
    ALIGNW --> PG
    ASRW --> PG
    TRW --> PG
    TTSW --> S3
    TTSW --> PG
    MIXW --> S3
    MIXW --> PG
    QAW --> PG

    API --> MON
    ORCH --> MON
    ASRW --> MON
    TTSW --> MON
    QAW --> MON
```

---

## 11. Data model

```mermaid
erDiagram
    PROJECT ||--o{ VIDEO : contains
    VIDEO ||--o{ JOB : has
    JOB ||--o{ SEGMENT : processes
    SEGMENT }o--|| SPEAKER : assigned_to
    SEGMENT ||--o{ VAD_REGION : derived_from
    SEGMENT ||--o{ TRANSCRIPTION : has
    SEGMENT ||--o{ TRANSLATION : has
    SEGMENT ||--o{ SYNTHESIS : has
    SEGMENT ||--o{ QA_ISSUE : may_raise
    VIDEO ||--o{ ASSET : stores
    JOB ||--o{ JOB_EVENT : logs

    PROJECT {
        uuid id
        string name
        datetime created_at
    }

    VIDEO {
        uuid id
        uuid project_id
        string source_path
        int duration_ms
        string source_language
    }

    JOB {
        uuid id
        uuid video_id
        string status
        float progress
        datetime created_at
    }

    VAD_REGION {
        uuid id
        uuid video_id
        int start_ms
        int end_ms
        float confidence
    }

    SEGMENT {
        uuid id
        uuid job_id
        uuid speaker_id
        int start_ms
        int end_ms
        string status
        float confidence
    }

    SPEAKER {
        uuid id
        string label
        string voice_profile
    }

    TRANSCRIPTION {
        uuid id
        uuid segment_id
        text source_text
        json word_timestamps
    }

    TRANSLATION {
        uuid id
        uuid segment_id
        text target_text
        text compact_text
        float estimated_duration_ms
    }

    SYNTHESIS {
        uuid id
        uuid segment_id
        string wav_path
        float actual_duration_ms
        float stretch_ratio
    }

    QA_ISSUE {
        uuid id
        uuid segment_id
        string issue_type
        string severity
    }

    ASSET {
        uuid id
        uuid video_id
        string type
        string path
    }

    JOB_EVENT {
        uuid id
        uuid job_id
        string event_type
        text payload
    }
```

---

## 12. Critical subflow: duration control

This is the most important logic loop in the whole system.

The largest technical risk is not ASR quality.  
It is temporal mismatch between translated text and available speaking time.

```mermaid
flowchart TD
    A[Source text plus target time window] --> B[Translate]
    B --> C[Estimate expected duration]
    C --> D{Estimated duration <= target duration?}

    D -- Yes --> E[Synthesize with Qwen TTS]
    D -- No --> F[Rewrite shorter translation]
    F --> C

    E --> G[Measure generated audio]
    G --> H{Real timing error within threshold?}

    H -- Yes --> I[Accept]
    H -- Small error --> J[Apply mild retiming]
    J --> I
    H -- Large error --> F
```

---

## 13. Critical subflow: QA

```mermaid
flowchart TD
    A[Synthesized segment] --> B[Compare target vs actual duration]
    B --> C[Detect overlaps]
    C --> D[Detect excessive gaps]
    D --> E[Check loudness]
    E --> F[Check clipping]
    F --> G[Check transcript and synthesis confidence]
    G --> H{Pass all rules?}
    H -- Yes --> I[QA OK]
    H -- No --> J[Flag for rewrite or regeneration]
```

---

## 14. Processing strategy for 2–3 hour videos

The system should not be designed around “one video job.”

It should be designed around a large collection of independently managed segments.

Typical rules:

- initial preprocessing over the full file,
- VAD creates speech regions,
- speech regions are grouped into manageable chunks,
- each chunk produces aligned utterances,
- each utterance becomes the unit for translation and TTS,
- failed utterances are retried without restarting the full job.

This makes the system:

- restartable,
- parallelizable,
- auditable,
- much cheaper to debug,
- compatible with manual intervention.

---

## 15. What the VAD changes in the architecture

With no VAD, the pipeline depends on fixed-size windows or rough chunking.  
That causes poorer speech boundary quality and more downstream repair.

With VAD, the system gains a better first approximation of speech timing before ASR.  
That improves:

- segmentation quality,
- pacing reconstruction,
- non-speech filtering,
- cost efficiency,
- synchronization reliability.

In other words, VAD reduces downstream correction pressure.

---

## 16. Recommended MVP

A practical first production-grade MVP should include:

- video upload,
- audio extraction,
- optional stem preparation,
- Silero VAD,
- speaker diarization,
- Qwen ASR,
- forced alignment,
- translation with target-duration constraints,
- Qwen TTS,
- light retiming,
- timeline assembly,
- QA checks,
- manual review interface for failed segments.

The manual review interface is not a luxury.  
It is part of a realistic production system.

---

## 17. Final engineering judgment

For long-form translated dubbing, the architecture should not rely on Qwen ASR and Qwen TTS alone.

It should use:

- **Silero VAD** for speech activity detection,
- **Qwen ASR** for transcription,
- **forced alignment** for timestamp precision,
- **duration-aware translation** for timing control,
- **Qwen TTS** for synthesis,
- **retiming and QA** for final synchronization quality.

That architecture is technically coherent, scalable to multi-hour videos, and much more likely to produce stable synchronization against picture.

---

## 18. Suggested next implementation step

The next sensible step is to convert this document into:

- a repository blueprint,
- service contracts and payload schemas,
- database migrations,
- queue task definitions,
- and a first worker implementation order.

A good implementation sequence would be:

1. ingest plus preprocessing,
2. VAD plus segmentation,
3. ASR plus alignment,
4. translation plus duration-control loop,
5. TTS plus retiming,
6. mixing plus export,
7. review UI plus QA tooling.
