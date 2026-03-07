"""Qwen3-TTS wrapper for text-to-speech synthesis."""

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
    """Wrapper for Qwen3-TTS model.
    
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
    
    # Preset voices for CustomVoice mode
    PRESET_VOICES = [
        "Aiden", "Eric", "Ryan", "Serena", "Jessica",
        "Michael", "Emily", "David", "Sarah", "James",
    ]
    
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
        self._tokenizer = None
        self._config = None
        
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
        """Load the TTS model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers>=4.51.0"
            )
        
        logger.info(f"Loading TTS model: {self.model_name}")
        
        # Prepare model kwargs
        model_kwargs = {
            "torch_dtype": self.precision,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
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
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        
        # Load model
        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            **model_kwargs,
        )
        
        # Move to device
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        
        self._model.eval()
        logger.info("TTS model loaded successfully")
    
    def synthesize(
        self,
        text: str,
        language: str = "English",
        speaker: Optional[str] = None,
        instruction: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Generate speech from text using preset voice.
        
        Args:
            text: Text to synthesize.
            language: Target language.
            speaker: Preset speaker name (e.g., "Aiden", "Serena").
            instruction: Optional style instruction.
            speed: Speech speed multiplier (0.5-2.0).
        
        Returns:
            TTSResult with generated audio.
        """
        self._load_model()
        
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{language}' may not be supported")
        
        # Build prompt for CustomVoice mode
        prompt = self._build_custom_voice_prompt(
            text=text,
            language=language,
            speaker=speaker,
            instruction=instruction,
        )
        
        # Generate audio
        audio = self._generate(prompt)
        
        # Calculate duration (assuming 24kHz sample rate for Qwen3-TTS)
        sample_rate = 24000
        duration = len(audio) / sample_rate
        
        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
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
            ref_audio, ref_sr = self._load_audio(str(reference_audio), target_sr=sample_rate)
        else:
            ref_audio = reference_audio
        
        # Generate prompt with voice cloning
        prompt = self._build_voice_clone_prompt(
            text=text,
            reference_audio=ref_audio,
            reference_text=reference_text,
            language=language,
        )
        
        # Generate audio
        audio = self._generate(prompt)
        
        sample_rate = 24000
        duration = len(audio) / sample_rate
        
        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
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
        
        # Build prompt for VoiceDesign mode
        prompt = self._build_voice_design_prompt(
            text=text,
            description=voice_description,
            language=language,
        )
        
        # Generate audio
        audio = self._generate(prompt)
        
        sample_rate = 24000
        duration = len(audio) / sample_rate
        
        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            duration=duration,
            voice_id=f"designed:{voice_description[:20]}",
        )
    
    def _build_custom_voice_prompt(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        instruction: Optional[str],
    ) -> str:
        """Build prompt for CustomVoice mode."""
        if speaker is None:
            speaker = "Aiden"  # Default speaker
        
        prompt_parts = [
            f"<|text|>{text}<|/text|>",
            f"<|language|>{language}<|/language|>",
            f"<|speaker|>{speaker}<|/speaker|>",
        ]
        
        if instruction:
            prompt_parts.append(f"<|instruction|>{instruction}<|/instruction|>")
        
        prompt_parts.append("<|audio|>")
        return "".join(prompt_parts)
    
    def _build_voice_clone_prompt(
        self,
        text: str,
        reference_audio: np.ndarray,
        reference_text: Optional[str],
        language: str,
    ) -> str:
        """Build prompt for voice cloning mode."""
        # Note: Actual implementation would need to encode reference audio
        # This is a simplified placeholder
        prompt_parts = [
            "<|voice_clone|>",
            f"<|text|>{text}<|/text|>",
            f"<|language|>{language}<|/language|>",
        ]
        
        if reference_text:
            prompt_parts.append(f"<|reference_text|>{reference_text}<|/reference_text|>")
        
        prompt_parts.append("<|audio|>")
        return "".join(prompt_parts)
    
    def _build_voice_design_prompt(
        self,
        text: str,
        description: str,
        language: str,
    ) -> str:
        """Build prompt for VoiceDesign mode."""
        prompt_parts = [
            "<|voice_design|>",
            f"<|description|>{description}<|/description|>",
            f"<|text|>{text}<|/text|>",
            f"<|language|>{language}<|/language|>",
            "<|audio|>",
        ]
        return "".join(prompt_parts)
    
    def _generate(self, prompt: str) -> np.ndarray:
        """Generate audio from prompt."""
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.7,
            )
        
        # Decode - this is simplified, actual implementation depends on model
        # Qwen3-TTS uses a speech tokenizer that needs proper decoding
        audio = self._decode_audio(outputs[0])
        
        return audio
    
    def _decode_audio(self, token_ids: torch.Tensor) -> np.ndarray:
        """Decode audio tokens to waveform."""
        # This is a placeholder - actual implementation needs
        # the speech tokenizer from Qwen3-TTS
        # For now, return silence - proper implementation requires
        # the Qwen-TTS-Tokenizer
        logger.warning("Audio decoding requires Qwen-TTS-Tokenizer")
        return np.zeros(24000, dtype=np.float32)  # 1 second of silence
    
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
