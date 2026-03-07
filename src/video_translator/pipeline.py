"""Main video translation pipeline orchestrator."""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from .config import Config, get_config
from .models.asr import QwenASR, ASRResult
from .models.tts import QwenTTS, TTSResult
from .models.aligner import QwenForcedAligner, AlignmentResult
from .processing.audio import AudioProcessor, extract_audio
from .processing.video import VideoProcessor, mux_audio_video
from .processing.subtitles import SubtitleGenerator, generate_srt

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


def get_language_name(code: str) -> str:
    """Convert language code to full language name for TTS."""
    # Check if already a full name (case-insensitive)
    for full_name in LANGUAGE_MAP.values():
        if code.lower() == full_name.lower():
            return full_name
    return LANGUAGE_MAP.get(code.lower(), code)


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
            srt_path = generate_srt(
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
    ) -> str:
        """Translate text from source to target language.
        
        Uses the NLLB (No Language Left Behind) model for translation.
        
        Args:
            text: Text to translate.
            source_language: Source language code.
            target_language: Target language code.
        
        Returns:
            Translated text.
        """
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        # Language code mapping for NLLB (BCP-47 codes)
        NLLB_LANGUAGE_MAP = {
            "spanish": "spa_Latn",
            "english": "eng_Latn",
            "chinese": "zho_Hans",
            "french": "fra_Latn",
            "german": "deu_Latn",
            "italian": "ita_Latn",
            "japanese": "jpn_Jpan",
            "korean": "kor_Hang",
            "portuguese": "por_Latn",
            "russian": "rus_Cyrl",
        }
        
        source_code = NLLB_LANGUAGE_MAP.get(source_language.lower(), "spa_Latn")
        target_code = NLLB_LANGUAGE_MAP.get(target_language.lower(), "eng_Latn")
        
        logger.info(f"Loading NLLB translation model...")
        model_name = "facebook/nllb-200-distilled-600M"
        
        try:
            # Initialize tokenizer with source language
            tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_code)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            
            # Generate translation with forced target language
            target_token_id = tokenizer.convert_tokens_to_ids(target_code)
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_length=512,
            )
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Translation: '{text}' -> '{translated}'")
            return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Fallback: return original text
            return text
    
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
        
        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Transcribe original audio
            logger.info("=" * 50)
            logger.info("Step 1: Transcribing original audio")
            transcription = self.transcribe(
                input_path,
                output_dir=temp_path,
                generate_srt=False,
            )
            logger.info(f"Detected language: {transcription.language}")
            
            # Step 2: Translate text
            logger.info("=" * 50)
            logger.info("Step 2: Translating text")
            translated_text = self.translate_text(
                transcription.text,
                source_language=transcription.language,
                target_language=target_language,
            )
            
            # Save translated text
            transcript_path = output_dir / f"{input_path.stem}_translated.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(translated_text)
            
            # Step 3: Generate speech in target language
            logger.info("=" * 50)
            logger.info("Step 3: Generating speech")
            audio_path = temp_path / "translated_audio.wav"
            
            if voice_clone and transcription.audio_path:
                tts_result = self.synthesize_speech(
                    text=translated_text,
                    output_path=audio_path,
                    language=get_language_name(target_language),
                    voice_clone=True,
                    reference_audio=transcription.audio_path,
                    reference_text=transcription.text,
                )
            else:
                tts_result = self.synthesize_speech(
                    text=translated_text,
                    output_path=audio_path,
                    language=get_language_name(target_language),
                    speaker=speaker,
                )
            
            # Step 4: Mux audio with video
            logger.info("=" * 50)
            logger.info("Step 4: Muxing audio with video")
            video_path = output_dir / f"{input_path.stem}_{target_language}.mp4"
            
            self.video_processor.replace_audio(
                input_path,
                audio_path,
                video_path,
            )
            
            # Step 5: Generate subtitles (optional)
            subtitle_path = None
            if generate_subtitles:
                logger.info("=" * 50)
                logger.info("Step 5: Generating subtitles")
                
                # For now, create simple subtitle from translated text
                # In production, align with audio for precise timing
                subtitle_path = output_dir / f"{input_path.stem}_{target_language}.srt"
                segments = [
                    {"start": 0, "end": 5, "text": translated_text[:200]}
                ]  # Placeholder
                generate_srt(segments, subtitle_path)
            
            return TranslationResult(
                video_path=video_path,
                audio_path=audio_path,
                transcript_path=transcript_path,
                subtitle_path=subtitle_path,
                original_language=transcription.language,
                target_language=target_language,
            )
    
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
