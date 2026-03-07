"""Qwen3-ASR wrapper for speech-to-text transcription."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """Result of ASR transcription."""
    
    text: str
    """Transcribed text."""
    
    language: str
    """Detected language code."""
    
    timestamps: List[Dict[str, Any]] = field(default_factory=list)
    """Word-level timestamps if available."""
    
    confidence: float = 1.0
    """Confidence score."""
    
    duration: float = 0.0
    """Audio duration in seconds."""


class QwenASR:
    """Wrapper for Qwen3-ASR model.
    
    Supports both 0.6B and 1.7B models with automatic language detection
    for 52 languages and dialects.
    
    Example:
        >>> asr = QwenASR(model_name="Qwen/Qwen3-ASR-1.7B")
        >>> result = asr.transcribe("audio.wav")
        >>> print(result.text)
        >>> print(result.timestamps)  # Word-level timestamps
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: Optional[str] = None,
        precision: str = "bf16",
        flash_attention: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize Qwen3-ASR model.
        
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
        self._processor = None
        
        logger.info(f"Initialized QwenASR with model: {model_name}")
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
        """Load the ASR model and processor."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers>=4.51.0"
            )
        
        logger.info(f"Loading ASR model: {self.model_name}")
        
        # Prepare model kwargs
        model_kwargs = {
            "torch_dtype": self.precision,
            "low_cpu_mem_usage": True,
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
        
        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        
        # Load model
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            **model_kwargs,
        )
        
        # Move to device
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        
        self._model.eval()
        logger.info("ASR model loaded successfully")
    
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        return_timestamps: bool = True,
        chunk_length: float = 30.0,
        batch_size: int = 16,
    ) -> ASRResult:
        """Transcribe audio to text.
        
        Args:
            audio: Path to audio file or numpy array of audio samples.
            sample_rate: Sample rate of audio (required if audio is array).
            language: Language code (auto-detected if None).
            return_timestamps: Whether to return word-level timestamps.
            chunk_length: Length of audio chunks in seconds.
            batch_size: Batch size for processing.
        
        Returns:
            ASRResult with transcribed text and optional timestamps.
        """
        self._load_model()
        
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio), target_sr=sample_rate)
        
        # Prepare input features
        inputs = self._processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate transcription
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=448,
                language=language,
                return_timestamps=return_timestamps,
            )
        
        # Decode transcription
        transcription = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        # Parse timestamps if available
        timestamps = []
        if return_timestamps and hasattr(self._processor, "token_decoder"):
            timestamps = self._parse_timestamps(generated_ids[0])
        
        # Calculate audio duration
        duration = len(audio) / sample_rate if len(audio) > 0 else 0.0
        
        return ASRResult(
            text=transcription,
            language=language or "auto",
            timestamps=timestamps,
            duration=duration,
        )
    
    def _load_audio(
        self,
        audio_path: str,
        target_sr: int = 16000,
    ) -> tuple[np.ndarray, int]:
        """Load audio file and resample if needed."""
        try:
            import librosa
        except ImportError:
            raise ImportError("Please install librosa: pip install librosa")
        
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            logger.debug(f"Resampled audio from {sr}Hz to {target_sr}Hz")
        
        return audio, target_sr
    
    def _parse_timestamps(
        self,
        token_ids: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Parse word-level timestamps from generated tokens."""
        # This is a simplified implementation
        # The actual parsing depends on the model's output format
        timestamps = []
        
        try:
            # Attempt to extract timestamps from the generation
            # This may need adjustment based on the actual model output
            for i, token in enumerate(token_ids):
                if token < 2:  # Timestamp tokens
                    # Extract timestamp information
                    pass
        except Exception as e:
            logger.warning(f"Could not parse timestamps: {e}")
        
        return timestamps
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ASR model unloaded")
