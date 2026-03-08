"""Main video translation pipeline orchestrator."""

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, get_config
from .models.asr import QwenASR
from .models.tts import QwenTTS, TTSResult
from .models.aligner import QwenForcedAligner, AlignmentResult
from .processing.audio import AudioProcessor
from .processing.video import VideoProcessor
from .processing.subtitles import SubtitleGenerator
from .processing.vad import SileroVAD, SpeechRegion
from .processing.qa import SegmentQA

logger = logging.getLogger(__name__)

# Language code mapping for TTS (ISO 639-1 to full language name)
LANGUAGE_MAP = {
    "es": "Spanish",
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
}

# Language code mapping for NLLB translation model (BCP-47 codes).
NLLB_LANGUAGE_MAP = {
    "spanish": "spa_Latn",
    "es": "spa_Latn",
    "spa": "spa_Latn",
    "english": "eng_Latn",
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "chinese": "zho_Hans",
    "zh": "zho_Hans",
    "zho": "zho_Hans",
    "french": "fra_Latn",
    "fr": "fra_Latn",
    "fra": "fra_Latn",
    "german": "deu_Latn",
    "de": "deu_Latn",
    "deu": "deu_Latn",
    "italian": "ita_Latn",
    "it": "ita_Latn",
    "ita": "ita_Latn",
    "japanese": "jpn_Jpan",
    "ja": "jpn_Jpan",
    "jpn": "jpn_Jpan",
    "korean": "kor_Hang",
    "ko": "kor_Hang",
    "kor": "kor_Hang",
    "portuguese": "por_Latn",
    "pt": "por_Latn",
    "por": "por_Latn",
    "russian": "rus_Cyrl",
    "ru": "rus_Cyrl",
    "rus": "rus_Cyrl",
}


def get_language_name(code: str) -> str:
    """Convert language code to full language name for TTS."""
    # Check if already a full name (case-insensitive)
    for full_name in LANGUAGE_MAP.values():
        if code.lower() == full_name.lower():
            return full_name
    return LANGUAGE_MAP.get(code.lower(), code)


def get_nllb_code(language: str, default: str = "eng_Latn") -> str:
    """Resolve human/code language input to an NLLB language code."""
    if not language:
        return default
    return NLLB_LANGUAGE_MAP.get(language.lower(), default)


@dataclass
class TranscriptionResult:
    """Result of transcription."""
    
    text: str
    """Transcribed text."""
    
    language: str
    """Detected language."""
    
    timestamps: List[Dict[str, Any]] = field(default_factory=list)
    """Word-level timestamps."""
    
    audio_path: Optional[Path] = None
    """Path to extracted audio file."""
    
    srt_path: Optional[Path] = None
    """Path to generated SRT file."""


@dataclass
class TranslationResult:
    """Result of video translation."""
    
    video_path: Path
    """Path to translated video."""
    
    audio_path: Path
    """Path to generated audio."""
    
    transcript_path: Path
    """Path to transcript file."""
    
    subtitle_path: Optional[Path] = None
    """Path to subtitle file (if generated)."""
    
    original_language: str = ""
    """Original detected language."""
    
    target_language: str = ""
    """Target translation language."""


@dataclass
class SegmentTranslationResult:
    """Per-segment translation and synthesis output."""

    segment_id: int
    start: float
    end: float
    source_text: str
    translated_text: str
    audio_path: Path
    actual_duration: float


@dataclass
class PreparedSegment:
    """Intermediate segment data with token estimate for translation progress."""

    segment_id: int
    start: float
    end: float
    target_duration: float
    audio_path: Path
    source_text: str
    source_language: str
    token_count: int


class VideoTranslator:
    """Main class for video translation pipeline.
    
    Provides methods for:
    - Audio transcription (ASR)
    - Text-to-speech synthesis (TTS)
    - Forced alignment for timestamps
    - Full video translation
    
    Example:
        >>> translator = VideoTranslator()
        >>> result = translator.transcribe("video.mp4")
        >>> print(result.text)
        
        >>> result = translator.translate_video(
        ...     "english_video.mp4",
        ...     target_language="spanish"
        ... )
        >>> print(f"Translated video: {result.video_path}")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        asr_model: Optional[str] = None,
        tts_model: Optional[str] = None,
        aligner_model: Optional[str] = None,
    ):
        """Initialize VideoTranslator.
        
        Args:
            config: Configuration object (uses default if None).
            asr_model: Override ASR model name.
            tts_model: Override TTS model name.
            aligner_model: Override aligner model name.
        """
        self.config = config or get_config()
        
        # Model configurations
        self.asr_model_name = asr_model or self.config.qwen_asr_model
        self.tts_model_name = tts_model or self.config.qwen_tts_model
        self.aligner_model_name = aligner_model or self.config.qwen_aligner_model
        
        # Lazy-loaded components
        self._asr: Optional[QwenASR] = None
        self._tts: Optional[QwenTTS] = None
        self._aligner: Optional[QwenForcedAligner] = None
        self._audio_processor: Optional[AudioProcessor] = None
        self._video_processor: Optional[VideoProcessor] = None
        self._subtitle_generator: Optional[SubtitleGenerator] = None
        self._vad: Optional[SileroVAD] = None
        self._segment_qa: Optional[SegmentQA] = None
        self._translation_model = None
        self._translation_model_device = "cpu"
        self._translation_tokenizers: Dict[str, Any] = {}
        self._translation_model_name = "facebook/nllb-200-distilled-600M"
        self._translation_total_tokens = 0
        self._translation_processed_tokens = 0
        
        logger.info("VideoTranslator initialized")
        logger.info(f"ASR Model: {self.asr_model_name}")
        logger.info(f"TTS Model: {self.tts_model_name}")
        logger.info(f"Aligner Model: {self.aligner_model_name}")
    
    @property
    def asr(self) -> QwenASR:
        """Get or create ASR model."""
        if self._asr is None:
            self._asr = QwenASR(
                model_name=self.asr_model_name,
                device=self.config.device,
                precision=self.config.precision,
                flash_attention=self.config.flash_attention,
                cache_dir=str(self.config.model_cache_dir),
                forced_aligner_model=self.aligner_model_name,
            )
        return self._asr
    
    @property
    def tts(self) -> QwenTTS:
        """Get or create TTS model."""
        if self._tts is None:
            self._tts = QwenTTS(
                model_name=self.tts_model_name,
                device=self.config.device,
                precision=self.config.precision,
                flash_attention=self.config.flash_attention,
                cache_dir=str(self.config.model_cache_dir),
            )
        return self._tts
    
    @property
    def aligner(self) -> QwenForcedAligner:
        """Get or create ForcedAligner model."""
        if self._aligner is None:
            self._aligner = QwenForcedAligner(
                model_name=self.aligner_model_name,
                device=self.config.device,
                precision=self.config.precision,
                flash_attention=self.config.flash_attention,
                cache_dir=str(self.config.model_cache_dir),
            )
        return self._aligner
    
    @property
    def audio_processor(self) -> AudioProcessor:
        """Get or create audio processor."""
        if self._audio_processor is None:
            self._audio_processor = AudioProcessor(
                sample_rate=self.config.audio_sample_rate,
                channels=self.config.audio_channels,
            )
        return self._audio_processor
    
    @property
    def video_processor(self) -> VideoProcessor:
        """Get or create video processor."""
        if self._video_processor is None:
            self._video_processor = VideoProcessor()
        return self._video_processor
    
    @property
    def subtitle_generator(self) -> SubtitleGenerator:
        """Get or create subtitle generator."""
        if self._subtitle_generator is None:
            self._subtitle_generator = SubtitleGenerator()
        return self._subtitle_generator

    @property
    def vad(self) -> SileroVAD:
        """Get or create VAD component."""
        if self._vad is None:
            self._vad = SileroVAD(
                threshold=self.config.vad_threshold,
                sampling_rate=self.config.audio_sample_rate,
                min_speech_duration_ms=self.config.vad_min_speech_duration_ms,
                min_silence_duration_ms=self.config.vad_min_silence_duration_ms,
            )
        return self._vad

    @property
    def segment_qa(self) -> SegmentQA:
        """Get or create segment QA validator."""
        if self._segment_qa is None:
            self._segment_qa = SegmentQA(
                max_duration_error_ratio=self.config.duration_error_tolerance
            )
        return self._segment_qa
    
    def transcribe(
        self,
        video_path: Path,
        output_dir: Optional[Path] = None,
        generate_srt: bool = True,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio from video file.
        
        Args:
            video_path: Path to input video file.
            output_dir: Directory for output files (uses config default if None).
            generate_srt: Whether to generate SRT subtitle file.
            language: Language code (auto-detected if None).
        
        Returns:
            TranscriptionResult with text and optional subtitles.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        output_dir = Path(output_dir) if output_dir else self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract audio
        logger.info(f"Extracting audio from {video_path.name}...")
        audio_info = self.audio_processor.extract_audio(
            video_path,
            output_path=output_dir / f"{video_path.stem}_audio.wav",
        )
        
        # Step 2: Transcribe audio
        logger.info("Transcribing audio...")
        asr_result = self.asr.transcribe(
            audio_info.path,
            sample_rate=self.config.audio_sample_rate,
            language=language,
            return_timestamps=True,
        )
        
        # Step 3: Generate SRT if requested
        srt_path = None
        if generate_srt and asr_result.timestamps:
            segments = [
                {
                    "start": ts.get("start", 0),
                    "end": ts.get("end", 0),
                    "text": ts.get("text", ""),
                }
                for ts in asr_result.timestamps
            ]
            srt_path = self.subtitle_generator.generate_srt(
                segments,
                output_dir / f"{video_path.stem}.srt",
            )
        
        # Save transcript
        transcript_path = output_dir / f"{video_path.stem}_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(asr_result.text)
        
        return TranscriptionResult(
            text=asr_result.text,
            language=asr_result.language,
            timestamps=asr_result.timestamps,
            audio_path=audio_info.path,
            srt_path=srt_path,
        )
    
    def synthesize_speech(
        self,
        text: str,
        output_path: Path,
        language: str = "English",
        speaker: Optional[str] = None,
        voice_clone: bool = False,
        reference_audio: Optional[Path] = None,
        reference_text: Optional[str] = None,
        voice_design: bool = False,
        voice_description: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize.
            output_path: Path for output audio file.
            language: Target language.
            speaker: Preset speaker name.
            voice_clone: Enable voice cloning mode.
            reference_audio: Reference audio for voice cloning.
            voice_design: Enable voice design mode.
            voice_description: Voice description for design mode.
        
        Returns:
            TTSResult with generated audio.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if voice_design and voice_description:
            logger.info(f"Synthesizing with voice design: {voice_description}")
            result = self.tts.synthesize_voice_design(
                text=text,
                voice_description=voice_description,
                language=language,
            )
        elif voice_clone and reference_audio:
            logger.info("Synthesizing with voice cloning")
            result = self.tts.synthesize_voice_clone(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                language=language,
            )
        else:
            logger.info(f"Synthesizing with speaker: {speaker}")
            result = self.tts.synthesize(
                text=text,
                language=language,
                speaker=speaker,
            )
        
        # Save audio to file
        import soundfile as sf
        sf.write(str(output_path), result.audio, result.sample_rate)
        logger.info(f"Audio saved to {output_path}")
        
        return result
    
    def align_audio_text(
        self,
        audio_path: Path,
        text: str,
        language: str = "English",
    ) -> AlignmentResult:
        """Align audio with text to get precise timestamps.
        
        Args:
            audio_path: Path to audio file.
            text: Text to align with audio.
            language: Language of the text.
        
        Returns:
            AlignmentResult with word-level timestamps.
        """
        logger.info("Aligning audio with text...")
        result = self.aligner.align(
            audio_path,
            text,
            language=language,
        )
        logger.info(f"Alignment complete: {len(result.segments)} segments")
        return result
    
    def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
        max_tokens: int = 800,
    ) -> str:
        """Translate text from source to target language.
        
        Uses the NLLB (No Language Left Behind) model for translation.
        For long texts, automatically splits into chunks and translates each chunk.
        
        Args:
            text: Text to translate.
            source_language: Source language code.
            target_language: Target language code.
            max_tokens: Maximum tokens per chunk (NLLB limit is 1024).
        
        Returns:
            Translated text.
        """
        import torch

        source_code = get_nllb_code(source_language, default="eng_Latn")
        target_code = get_nllb_code(target_language, default="eng_Latn")
        if source_code == target_code:
            return text

        try:
            model, tokenizer, device = self._get_translation_backend(source_code)

            # Check if text needs chunking
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_length = inputs["input_ids"].shape[1]
            target_token_id = tokenizer.convert_tokens_to_ids(target_code)

            if input_length <= max_tokens:
                # Translate as single chunk
                logger.info(f"Translating {input_length} tokens (single chunk)")
                device_inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = model.generate(
                        **device_inputs,
                        forced_bos_token_id=target_token_id,
                        max_length=512,
                    )
                translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Translation: '{text[:50]}...' -> '{translated[:50]}...'")
                return translated
            else:
                # Split into chunks and translate each
                logger.info(f"Translating {input_length} tokens (chunked, max {max_tokens} per chunk)")
                return self._translate_chunked(
                    text, tokenizer, model, source_code, target_code, device, max_tokens
                )
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Fallback: return original text
            return text
    
    def _translate_chunked(
        self,
        text: str,
        tokenizer,
        model,
        source_code: str,
        target_code: str,
        device: str,
        max_tokens: int = 800,
    ) -> str:
        """Translate long text by splitting into chunks.
        
        Args:
            text: Text to translate.
            tokenizer: NLLB tokenizer.
            model: NLLB model.
            source_code: Source language code.
            target_code: Target language code.
            max_tokens: Maximum tokens per chunk.
        
        Returns:
            Translated text.
        """
        import torch

        # Split text into sentences for better chunking
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            # Estimate tokens for this sentence
            sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
            
            if current_token_count + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_token_count = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_token_count += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Translate each chunk
        target_token_id = tokenizer.convert_tokens_to_ids(target_code)
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Translating chunk {i+1}/{len(chunks)} ({len(tokenizer.encode(chunk, add_special_tokens=False))} tokens)")
            
            inputs = tokenizer(chunk, return_tensors="pt", padding=True)
            device_inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(
                    **device_inputs,
                    forced_bos_token_id=target_token_id,
                    max_length=512,
                )
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_chunks.append(translated)
            logger.info(f"Chunk {i+1}: '{chunk[:30]}...' -> '{translated[:30]}...'")
        
        # Combine translated chunks
        combined = ' '.join(translated_chunks)
        logger.info(f"Combined translation length: {len(combined)} characters")
        return combined

    def _reset_translation_progress(self, total_tokens: int) -> None:
        """Initialize translation token progress counters."""
        self._translation_total_tokens = max(0, int(total_tokens))
        self._translation_processed_tokens = 0

    def _advance_translation_progress(self, processed_tokens: int) -> None:
        """Advance token progress and emit an updated progress log line."""
        self._translation_processed_tokens += max(0, int(processed_tokens))
        if self._translation_total_tokens > 0:
            self._translation_processed_tokens = min(
                self._translation_processed_tokens,
                self._translation_total_tokens,
            )
        self._log_translation_progress()

    def _log_translation_progress(self) -> None:
        """Log processed/total translation tokens with percentage."""
        if self._translation_total_tokens <= 0:
            logger.info("Translation token progress: 0/0 (0.0%%)")
            return

        percentage = (
            100.0 * self._translation_processed_tokens / self._translation_total_tokens
        )
        logger.info(
            "Translation token progress: %d/%d (%.1f%%)",
            self._translation_processed_tokens,
            self._translation_total_tokens,
            percentage,
        )

    def _count_translation_tokens(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> int:
        """Count source translation tokens for progress tracking."""
        source_code = get_nllb_code(source_language, default="eng_Latn")
        target_code = get_nllb_code(target_language, default="eng_Latn")
        if source_code == target_code:
            return 0

        clean = (text or "").strip()
        if not clean:
            return 0

        try:
            _, tokenizer, _ = self._get_translation_backend(source_code)
            token_count = len(tokenizer.encode(clean, add_special_tokens=False))
            return max(1, token_count)
        except Exception:
            return max(1, len(clean.split()))
    
    def translate_text_with_timestamps(
        self,
        timestamps: list,
        source_language: str,
        target_language: str,
        max_tokens: int = 800,
    ) -> dict:
        """Translate text with timestamp preservation for long videos.
        
        Splits transcript into chunks based on timestamps, translates each chunk,
        and returns both the full text and segmented translations with timing.
        
        Args:
            timestamps: List of timestamp segments from ASR.
            source_language: Source language code.
            target_language: Target language code.
            max_tokens: Maximum tokens per chunk.
        
        Returns:
            Dictionary with 'full_text' and 'segments' (translated with timing).
        """
        import torch

        source_code = get_nllb_code(source_language, default="eng_Latn")
        target_code = get_nllb_code(target_language, default="eng_Latn")
        if source_code == target_code:
            return {
                "full_text": " ".join(ts.get("text", "") for ts in timestamps).strip(),
                "segments": timestamps,
            }
        model, tokenizer, device = self._get_translation_backend(source_code)
        target_token_id = tokenizer.convert_tokens_to_ids(target_code)
        
        # Group timestamps into chunks
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for ts in timestamps:
            text = ts.get('text', '')
            sentence_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            
            if current_token_count + sentence_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [ts]
                current_token_count = sentence_tokens
            else:
                current_chunk.append(ts)
                current_token_count += sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"Split transcript into {len(chunks)} chunks for translation")
        
        # Translate each chunk
        translated_segments = []
        full_translations = []
        
        for i, chunk in enumerate(chunks):
            # Combine text from this chunk
            chunk_text = ' '.join(ts.get('text', '') for ts in chunk)
            
            logger.info(f"Translating chunk {i+1}/{len(chunks)} ({len(tokenizer.encode(chunk_text, add_special_tokens=False))} tokens)")
            
            # Translate
            inputs = tokenizer(chunk_text, return_tensors="pt", padding=True)
            device_inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(
                    **device_inputs,
                    forced_bos_token_id=target_token_id,
                    max_length=512,
                )
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_translations.append(translated)
            
            # For word-level segments, we need to split the translated text
            # Simple approach: distribute translation proportionally
            if len(chunk) == 1:
                # Single segment - use full translation
                translated_segments.append({
                    'start': chunk[0]['start'],
                    'end': chunk[0]['end'],
                    'text': translated,
                })
            else:
                # Multiple segments - split translation by word count
                original_words = chunk_text.split()
                translated_words = translated.split()
                
                word_ratio = len(translated_words) / len(original_words) if original_words else 1
                
                start_idx = 0
                for j, ts in enumerate(chunk):
                    ts_words = ts.get('text', '').split()
                    translated_count = max(1, int(len(ts_words) * word_ratio))
                    
                    # Get words for this segment
                    segment_words = translated_words[start_idx:start_idx + translated_count]
                    segment_text = ' '.join(segment_words)
                    
                    translated_segments.append({
                        'start': ts['start'],
                        'end': ts['end'],
                        'text': segment_text,
                    })
                    
                    start_idx += translated_count
        
        full_text = ' '.join(full_translations)
        logger.info(f"Translation complete: {len(full_text)} characters")
        
        return {
            'full_text': full_text,
            'segments': translated_segments,
        }

    def _get_translation_backend(self, source_code: str):
        """Load and cache the NLLB model/tokenizer once per pipeline run."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        def _resolve_device() -> str:
            configured = (self.config.device or "auto").lower()
            if configured == "cuda":
                return "cuda" if torch.cuda.is_available() else "cpu"
            if configured == "mps":
                mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                return "mps" if mps_ok else "cpu"
            if configured == "cpu":
                return "cpu"
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        cache_dir = str(self.config.model_cache_dir) if self.config.model_cache_dir else None

        if self._translation_model is None:
            device = _resolve_device()
            logger.info("Loading NLLB translation model on %s...", device)

            load_kwargs: Dict[str, Any] = {}
            if cache_dir:
                load_kwargs["cache_dir"] = cache_dir

            if device == "cuda":
                precision = (self.config.precision or "fp16").lower()
                if precision == "bf16":
                    load_kwargs["dtype"] = torch.bfloat16
                elif precision == "fp32":
                    load_kwargs["dtype"] = torch.float32
                else:
                    load_kwargs["dtype"] = torch.float16

            try:
                self._translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._translation_model_name,
                    **load_kwargs,
                )
                self._translation_model = self._translation_model.to(device)
                self._translation_model.eval()
                self._translation_model_device = device
            except Exception as exc:
                if device == "cpu":
                    raise
                logger.warning(
                    "NLLB load on %s failed (%s). Falling back to CPU.",
                    device,
                    exc,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                load_kwargs.pop("dtype", None)
                self._translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._translation_model_name,
                    **load_kwargs,
                )
                self._translation_model = self._translation_model.to("cpu")
                self._translation_model.eval()
                self._translation_model_device = "cpu"

        if source_code not in self._translation_tokenizers:
            tokenizer_kwargs: Dict[str, Any] = {"src_lang": source_code}
            if cache_dir:
                tokenizer_kwargs["cache_dir"] = cache_dir
            self._translation_tokenizers[source_code] = AutoTokenizer.from_pretrained(
                self._translation_model_name,
                **tokenizer_kwargs,
            )

        return (
            self._translation_model,
            self._translation_tokenizers[source_code],
            self._translation_model_device,
        )
    
    def translate_video(
        self,
        input_path: Path,
        target_language: str,
        output_dir: Optional[Path] = None,
        voice_clone: bool = True,
        generate_subtitles: bool = True,
        speaker: Optional[str] = None,
    ) -> TranslationResult:
        """Full video translation pipeline.
        
        Args:
            input_path: Path to input video file.
            target_language: Target language for translation.
            output_dir: Directory for output files.
            voice_clone: Clone original speaker's voice.
            generate_subtitles: Generate SRT subtitles.
            speaker: Preset speaker (if not voice cloning).
        
        Returns:
            TranslationResult with paths to all output files.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Video file not found: {input_path}")
        
        output_dir = Path(output_dir) if output_dir else self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_language_name = get_language_name(target_language)
        video_info = self.video_processor.get_video_info(input_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            logger.info("=" * 50)
            logger.info("Step 1: Extracting source audio")
            source_audio_info = self.audio_processor.extract_audio(
                input_path,
                output_path=temp_path / f"{input_path.stem}_source.wav",
                sample_rate=self.config.audio_sample_rate,
                channels=self.config.audio_channels,
            )
            source_audio_path = source_audio_info.path

            logger.info("=" * 50)
            logger.info("Step 2: Running VAD and building segments")
            vad_regions: List[SpeechRegion] = []
            if self.config.use_vad:
                vad_regions = self.vad.detect(source_audio_path)
                logger.info("VAD regions detected: %d", len(vad_regions))

            regions = self._build_processing_regions(vad_regions, video_info.duration)
            logger.info("Processing regions prepared: %d", len(regions))

            logger.info("=" * 50)
            logger.info("Step 3: Per-segment ASR -> translation -> TTS")
            prepared_segments: List[PreparedSegment] = []
            segment_results: List[SegmentTranslationResult] = []
            detected_source_language = "auto"

            segment_jobs: List[tuple[int, SpeechRegion, float, Path]] = []
            for idx, region in enumerate(regions):
                target_duration = max(0.0, region.end - region.start)
                if target_duration < self.config.min_segment_duration:
                    continue

                segment_audio_path = temp_path / f"segment_{idx:05d}.wav"
                segment_jobs.append((idx, region, target_duration, segment_audio_path))

            if not segment_jobs:
                raise RuntimeError("No translated speech segments were generated")

            configured_workers = int(getattr(self.config, "segment_extract_workers", 0))
            if configured_workers <= 0:
                cpu_count = os.cpu_count() or 1
                extract_workers = min(8, max(1, cpu_count - 1))
            else:
                extract_workers = max(1, configured_workers)
            extract_workers = min(extract_workers, len(segment_jobs))

            logger.info(
                "Extracting %d segments with %d CPU worker(s)",
                len(segment_jobs),
                extract_workers,
            )
            if extract_workers == 1:
                for _, region, _, segment_audio_path in segment_jobs:
                    self.audio_processor.extract_segment(
                        input_path=source_audio_path,
                        output_path=segment_audio_path,
                        start=region.start,
                        end=region.end,
                        sample_rate=self.config.audio_sample_rate,
                        channels=self.config.audio_channels,
                    )
            else:
                with ThreadPoolExecutor(max_workers=extract_workers) as executor:
                    future_to_segment = {
                        executor.submit(
                            self.audio_processor.extract_segment,
                            input_path=source_audio_path,
                            output_path=segment_audio_path,
                            start=region.start,
                            end=region.end,
                            sample_rate=self.config.audio_sample_rate,
                            channels=self.config.audio_channels,
                        ): (idx, segment_audio_path)
                        for idx, region, _, segment_audio_path in segment_jobs
                    }
                    for future in as_completed(future_to_segment):
                        idx, segment_audio_path = future_to_segment[future]
                        try:
                            future.result()
                        except Exception as exc:
                            raise RuntimeError(
                                f"Failed extracting segment {idx} ({segment_audio_path.name}): {exc}"
                            ) from exc

            for idx, region, target_duration, segment_audio_path in segment_jobs:
                asr_result = self.asr.transcribe(
                    segment_audio_path,
                    sample_rate=self.config.audio_sample_rate,
                    return_timestamps=True,
                )
                source_text = (asr_result.text or "").strip()
                if not source_text:
                    continue

                detected_source_language = asr_result.language or detected_source_language
                source_language = asr_result.language or detected_source_language

                # Alignment is best-effort in this flow; translation proceeds regardless.
                try:
                    self.aligner.align(
                        segment_audio_path,
                        source_text,
                        language=get_language_name(source_language),
                    )
                except Exception as exc:
                    logger.debug("Alignment skipped for segment %d: %s", idx, exc)

                token_count = self._count_translation_tokens(
                    text=source_text,
                    source_language=source_language,
                    target_language=target_language,
                )
                prepared_segments.append(
                    PreparedSegment(
                        segment_id=idx,
                        start=region.start,
                        end=region.end,
                        target_duration=target_duration,
                        audio_path=segment_audio_path,
                        source_text=source_text,
                        source_language=source_language,
                        token_count=token_count,
                    )
                )

            if not prepared_segments:
                raise RuntimeError("No translated speech segments were generated")

            total_tokens = sum(segment.token_count for segment in prepared_segments)
            self._reset_translation_progress(total_tokens=total_tokens)
            if self._translation_total_tokens > 0:
                logger.info(
                    "Translation token budget: %d total tokens across %d segments",
                    self._translation_total_tokens,
                    len(prepared_segments),
                )
            self._log_translation_progress()

            for segment in prepared_segments:
                translated_text = self.translate_text(
                    text=segment.source_text,
                    source_language=segment.source_language,
                    target_language=target_language,
                )
                fitted_text = self._fit_translation_to_duration(
                    translated_text, target_duration=segment.target_duration
                )

                final_audio_path, actual_duration, final_text = self._synthesize_segment_with_fit(
                    segment_id=segment.segment_id,
                    text=fitted_text,
                    language=target_language_name,
                    target_duration=segment.target_duration,
                    output_dir=temp_path,
                    voice_clone=voice_clone,
                    reference_audio=segment.audio_path if voice_clone else None,
                    reference_text=segment.source_text if voice_clone else None,
                    speaker=speaker,
                )

                segment_results.append(
                    SegmentTranslationResult(
                        segment_id=segment.segment_id,
                        start=segment.start,
                        end=segment.end,
                        source_text=segment.source_text,
                        translated_text=final_text,
                        audio_path=final_audio_path,
                        actual_duration=actual_duration,
                    )
                )
                self._advance_translation_progress(segment.token_count)

            logger.info("=" * 50)
            logger.info("Step 4: Timeline assembly and muxing")
            final_audio_path = output_dir / f"{input_path.stem}_{target_language}.wav"
            self.audio_processor.assemble_timeline(
                segments=[
                    {"audio_path": seg.audio_path, "start": seg.start}
                    for seg in segment_results
                ],
                output_path=final_audio_path,
                total_duration=video_info.duration,
                sample_rate=self.config.audio_sample_rate,
            )

            video_path = output_dir / f"{input_path.stem}_{target_language}.mp4"
            self.video_processor.replace_audio(
                input_path,
                final_audio_path,
                video_path,
                audio_delay=0.0,
            )

            logger.info("=" * 50)
            logger.info("Step 5: Writing transcript, subtitles, and QA report")
            translated_text = " ".join(seg.translated_text for seg in segment_results).strip()
            transcript_path = output_dir / f"{input_path.stem}_translated.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(translated_text)

            subtitle_path = None
            if generate_subtitles:
                subtitle_path = output_dir / f"{input_path.stem}_{target_language}.srt"
                subtitle_segments = [
                    {"start": seg.start, "end": seg.end, "text": seg.translated_text}
                    for seg in segment_results
                ]
                subtitle_path = self.subtitle_generator.generate_srt(
                    subtitle_segments, subtitle_path
                )

            self._write_segment_report(
                output_path=output_dir / f"{input_path.stem}_{target_language}_segments.json",
                segment_results=segment_results,
            )

            return TranslationResult(
                video_path=video_path,
                audio_path=final_audio_path,
                transcript_path=transcript_path,
                subtitle_path=subtitle_path,
                original_language=detected_source_language,
                target_language=target_language,
            )

    def _build_processing_regions(
        self,
        vad_regions: List[SpeechRegion],
        total_duration: float,
    ) -> List[SpeechRegion]:
        """Merge and split VAD regions into manageable processing chunks."""
        if total_duration <= 0:
            return []

        fallback_regions = self._build_fixed_timeline_regions(total_duration)

        if not vad_regions:
            return fallback_regions

        ordered = sorted(vad_regions, key=lambda r: r.start)
        grouped: List[SpeechRegion] = []
        max_len = self.config.max_segment_duration
        merge_gap = self.config.vad_min_silence_duration_ms / 1000.0

        current_start = ordered[0].start
        current_end = ordered[0].end
        current_conf = ordered[0].confidence

        for region in ordered[1:]:
            gap = region.start - current_end
            proposed_len = region.end - current_start
            if gap <= merge_gap and proposed_len <= max_len:
                current_end = max(current_end, region.end)
                current_conf = max(current_conf, region.confidence)
            else:
                grouped.append(
                    SpeechRegion(
                        start=max(0.0, current_start),
                        end=min(total_duration, current_end),
                        confidence=current_conf,
                    )
                )
                current_start, current_end, current_conf = (
                    region.start,
                    region.end,
                    region.confidence,
                )

        grouped.append(
            SpeechRegion(
                start=max(0.0, current_start),
                end=min(total_duration, current_end),
                confidence=current_conf,
            )
        )

        # Split very long regions by max segment duration.
        split_regions: List[SpeechRegion] = []
        for region in grouped:
            duration = region.end - region.start
            if duration <= max_len:
                split_regions.append(region)
                continue
            start = region.start
            while start < region.end:
                end = min(region.end, start + max_len)
                split_regions.append(
                    SpeechRegion(start=start, end=end, confidence=region.confidence)
                )
                start = end

        filtered = [
            region
            for region in split_regions
            if (region.end - region.start) >= self.config.min_segment_duration
        ]
        if filtered:
            return filtered

        logger.warning(
            "VAD produced %d regions but none met min_segment_duration=%.2fs; "
            "falling back to fixed timeline chunks.",
            len(vad_regions),
            self.config.min_segment_duration,
        )
        return fallback_regions

    def _build_fixed_timeline_regions(self, total_duration: float) -> List[SpeechRegion]:
        """Build fixed-size timeline chunks as a resilient fallback."""
        if total_duration <= 0:
            return []

        max_len = max(
            float(self.config.max_segment_duration),
            float(self.config.min_segment_duration),
            0.1,
        )

        regions: List[SpeechRegion] = []
        start = 0.0
        while start < total_duration:
            end = min(total_duration, start + max_len)
            if end > start:
                regions.append(SpeechRegion(start=start, end=end, confidence=1.0))
            start = end

        return regions

    def _fit_translation_to_duration(
        self,
        text: str,
        target_duration: float,
        chars_per_second: float = 14.0,
    ) -> str:
        """Compact text heuristically to better match the target timing window."""
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return cleaned

        if target_duration <= 0:
            return cleaned

        max_chars = max(20, int(target_duration * chars_per_second))
        if len(cleaned) <= max_chars:
            return cleaned

        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        compact_parts: List[str] = []
        current_len = 0
        for sentence in sentences:
            next_len = current_len + len(sentence) + (1 if compact_parts else 0)
            if next_len > max_chars:
                break
            compact_parts.append(sentence)
            current_len = next_len

        if compact_parts:
            return " ".join(compact_parts)

        trimmed = cleaned[:max_chars]
        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]
        return trimmed.strip()

    def _synthesize_segment_with_fit(
        self,
        segment_id: int,
        text: str,
        language: str,
        target_duration: float,
        output_dir: Path,
        voice_clone: bool,
        reference_audio: Optional[Path],
        reference_text: Optional[str],
        speaker: Optional[str],
    ) -> tuple[Path, float, str]:
        """Synthesize one segment and fit output timing with retry/retiming."""
        candidate_text = text
        fallback_audio_path = output_dir / f"segment_{segment_id:05d}_tts.wav"
        fallback_duration = 0.0

        attempts = max(0, self.config.max_translation_retries)
        for attempt in range(attempts + 1):
            base_audio_path = output_dir / f"segment_{segment_id:05d}_tts_try{attempt}.wav"
            self.synthesize_speech(
                text=candidate_text,
                output_path=base_audio_path,
                language=language,
                speaker=speaker,
                voice_clone=voice_clone and reference_audio is not None,
                reference_audio=reference_audio,
                reference_text=reference_text,
            )
            base_info = self.audio_processor.get_audio_info(base_audio_path)
            fallback_audio_path = base_audio_path
            fallback_duration = base_info.duration

            if target_duration <= 0:
                return base_audio_path, base_info.duration, candidate_text

            ratio = base_info.duration / target_duration
            if abs(1.0 - ratio) <= self.config.duration_error_tolerance:
                return base_audio_path, base_info.duration, candidate_text

            mild_min = 1.0 / self.config.max_retiming_ratio
            mild_max = self.config.max_retiming_ratio
            if mild_min <= ratio <= mild_max:
                stretched_path = output_dir / f"segment_{segment_id:05d}_retimed_try{attempt}.wav"
                try:
                    stretched = self.audio_processor.time_stretch_to_duration(
                        input_path=base_audio_path,
                        output_path=stretched_path,
                        target_duration=target_duration,
                        max_stretch_ratio=self.config.max_retiming_ratio,
                    )
                    return stretched_path, stretched.duration, candidate_text
                except Exception as exc:
                    logger.debug(
                        "Retiming failed for segment %d attempt %d: %s",
                        segment_id,
                        attempt,
                        exc,
                    )

            if attempt < attempts:
                shrink_target = max(
                    self.config.min_segment_duration,
                    target_duration * (0.9 - 0.1 * attempt),
                )
                candidate_text = self._fit_translation_to_duration(
                    candidate_text, target_duration=shrink_target
                )

        return fallback_audio_path, fallback_duration, candidate_text

    def _write_segment_report(
        self,
        output_path: Path,
        segment_results: List[SegmentTranslationResult],
    ) -> None:
        """Write segment details and QA findings for manual review."""
        import json

        payload = {
            "segments": [
                {
                    "segment_id": seg.segment_id,
                    "start": seg.start,
                    "end": seg.end,
                    "source_text": seg.source_text,
                    "translated_text": seg.translated_text,
                    "audio_path": str(seg.audio_path),
                    "actual_duration": seg.actual_duration,
                }
                for seg in segment_results
            ],
            "qa_issues": [
                {
                    "segment_index": issue.segment_index,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "message": issue.message,
                }
                for issue in self.segment_qa.validate(
                    [
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "actual_duration": seg.actual_duration,
                            "audio_path": str(seg.audio_path),
                        }
                        for seg in segment_results
                    ]
                )
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    
    def unload_models(self) -> None:
        """Unload all models from memory."""
        if self._asr:
            self._asr.unload()
        if self._tts:
            self._tts.unload()
        if self._aligner:
            self._aligner.unload()
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("All models unloaded")
