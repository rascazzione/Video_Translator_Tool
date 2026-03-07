"""Qwen3-TTS wrapper for text-to-speech synthesis using official qwen-tts package."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result of TTS synthesis."""
    
    audio: np.ndarray
    """Generated audio samples."""
    
    sample_rate: int
    """Sample rate of generated audio."""
    
    duration: float
    """Audio duration in seconds."""
    
    voice_id: Optional[str] = None
    """Voice identifier used."""


class QwenTTS:
    """Wrapper for Qwen3-TTS model using the official qwen-tts package.
    
    Supports:
    - Base TTS with preset voices
    - Voice cloning from reference audio
    - Voice design from natural language descriptions
    - Custom voice with instructions
    
    Example:
        >>> tts = QwenTTS(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        >>> result = tts.synthesize("Hello, world!", language="English")
        >>> import soundfile as sf
        >>> sf.write("output.wav", result.audio, result.sample_rate)
    """
    
    # Preset voices for CustomVoice mode (from official docs)
    PRESET_VOICES = {
        "Vivian": "Bright, slightly edgy young female voice (Chinese)",
        "Serena": "Warm, gentle young female voice (Chinese)",
        "Uncle_Fu": "Seasoned male voice with low, mellow timbre (Chinese)",
        "Dylan": "Youthful Beijing male voice (Chinese Beijing Dialect)",
        "Eric": "Lively Chengdu male voice (Chinese Sichuan Dialect)",
        "Ryan": "Dynamic male voice with strong rhythmic drive (English)",
        "Aiden": "Sunny American male voice with clear midrange (English)",
        "Ono_Anna": "Playful Japanese female voice (Japanese)",
        "Sohee": "Warm Korean female voice with rich emotion (Korean)",
    }
    
    # Supported languages
    SUPPORTED_LANGUAGES = [
        "Chinese", "English", "Japanese", "Korean",
        "French", "German", "Spanish", "Portuguese",
        "Russian", "Italian",
    ]
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: Optional[str] = None,
        precision: str = "bf16",
        flash_attention: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize Qwen3-TTS model.
        
        Args:
            model_name: HuggingFace model ID.
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto).
            precision: Model precision ('bf16', 'fp16', 'fp32').
            flash_attention: Enable FlashAttention 2 for faster inference.
            cache_dir: Directory to cache model files.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = self._get_device(device)
        self.precision = self._get_dtype(precision)
        self.flash_attention = flash_attention
        
        self._model = None
        
        logger.info(f"Initialized QwenTTS with model: {model_name}")
        logger.info(f"Device: {self.device}, Precision: {self.precision}")
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine the best available device."""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _get_dtype(self, precision: str) -> torch.dtype:
        """Convert precision string to torch dtype."""
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return dtype_map.get(precision, torch.float32)
    
    def _load_model(self) -> None:
        """Load the TTS model using the official qwen-tts package."""
        if self._model is not None:
            return
        
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "Please install qwen-tts: pip install -U qwen-tts"
            )
        
        logger.info(f"Loading TTS model: {self.model_name}")
        
        # Prepare model kwargs for qwen-tts package
        model_kwargs = {
            "device_map": self.device,
            "dtype": self.precision,
        }
        
        # Add cache_dir if specified
        if self.cache_dir:
            model_kwargs["cache_dir"] = self.cache_dir
        
        # Enable FlashAttention 2 if requested and available
        if self.device == "cuda" and self.flash_attention:
            try:
                from flash_attn import flash_attn_func  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("FlashAttention 2 enabled")
            except ImportError:
                logger.warning("FlashAttention 2 not available, using sdpa")
                model_kwargs["attn_implementation"] = "sdpa"
        else:
            model_kwargs["attn_implementation"] = "sdpa"
        
        # Load model using official qwen-tts package
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        
        logger.info("TTS model loaded successfully using qwen-tts package")
    
    def synthesize(
        self,
        text: str,
        language: str = "English",
        speaker: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> TTSResult:
        """Generate speech from text using preset voice (CustomVoice mode).
        
        Args:
            text: Text to synthesize.
            language: Target language.
            speaker: Preset speaker name (e.g., "Aiden", "Serena", "Vivian").
            instruction: Optional style instruction.
        
        Returns:
            TTSResult with generated audio.
        """
        self._load_model()
        
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{language}' may not be supported")
        
        # Use default speaker if none provided
        if speaker is None:
            speaker = "Aiden"
        
        # Generate using official qwen-tts API
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruction or "",
        )
        
        audio = wavs[0]
        duration = len(audio) / sr
        
        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=duration,
            voice_id=speaker,
        )
    
    def synthesize_voice_clone(
        self,
        text: str,
        reference_audio: Union[str, Path, np.ndarray],
        reference_text: Optional[str] = None,
        sample_rate: int = 24000,
        language: str = "English",
    ) -> TTSResult:
        """Generate speech by cloning voice from reference audio.
        
        Args:
            text: Text to synthesize.
            reference_audio: Path to reference audio or audio array.
            reference_text: Transcript of reference audio (optional).
            sample_rate: Sample rate of reference audio.
            language: Target language.
        
        Returns:
            TTSResult with cloned voice audio.
        """
        self._load_model()
        
        # Load reference audio if path
        if isinstance(reference_audio, (str, Path)):
            ref_audio = str(reference_audio)
        else:
            ref_audio = (reference_audio, sample_rate)
        
        # Generate using official qwen-tts API
        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=reference_text,
        )
        
        audio = wavs[0]
        duration = len(audio) / sr
        
        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=duration,
            voice_id="cloned",
        )
    
    def synthesize_voice_design(
        self,
        text: str,
        voice_description: str,
        language: str = "English",
    ) -> TTSResult:
        """Generate speech with a voice designed from description.
        
        Args:
            text: Text to synthesize.
            voice_description: Natural language voice description.
            language: Target language.
        
        Returns:
            TTSResult with generated audio.
        
        Example:
            >>> result = tts.synthesize_voice_design(
            ...     "Ahoy matey!",
            ...     "gruff old pirate voice with rough texture"
            ... )
        """
        self._load_model()
        
        # Generate using official qwen-tts API
        wavs, sr = self._model.generate_voice_design(
            text=text,
            language=language,
            instruct=voice_description,
        )
        
        audio = wavs[0]
        duration = len(audio) / sr
        
        return TTSResult(
            audio=audio,
            sample_rate=sr,
            duration=duration,
            voice_id=f"designed:{voice_description[:20]}",
        )
    
    def _load_audio(
        self,
        audio_path: str,
        target_sr: int = 24000,
    ) -> tuple[np.ndarray, int]:
        """Load audio file and resample if needed."""
        try:
            import librosa
        except ImportError:
            raise ImportError("Please install librosa: pip install librosa")
        
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio, target_sr
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("TTS model unloaded")
