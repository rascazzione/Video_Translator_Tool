"""Qwen3-ASR wrapper for speech-to-text transcription using official qwen-asr package."""

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
    """Wrapper for Qwen3-ASR model using the official qwen-asr package.
    
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
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        forced_aligner_model: Optional[str] = None,
    ):
        """Initialize Qwen3-ASR model.
        
        Args:
            model_name: HuggingFace model ID.
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto).
            precision: Model precision ('bf16', 'fp16', 'fp32').
            flash_attention: Enable FlashAttention 2 for faster inference.
            cache_dir: Directory to cache model files.
            max_inference_batch_size: Batch size limit for inference. -1 means unlimited.
            max_new_tokens: Maximum number of tokens to generate.
            forced_aligner_model: HuggingFace model ID for forced aligner (e.g., "Qwen/Qwen3-ForcedAligner-0.6B").
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = self._get_device(device)
        self.precision = self._get_dtype(precision)
        self.flash_attention = flash_attention
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.forced_aligner_model = forced_aligner_model
        
        self._model = None
        
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
        """Load the ASR model using the official qwen-asr package."""
        if self._model is not None:
            return
        
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "Please install qwen-asr: pip install -U qwen-asr"
            )
        
        logger.info(f"Loading ASR model: {self.model_name}")
        
        # Prepare model kwargs for qwen-asr package
        model_kwargs = {
            "dtype": self.precision,
            "device_map": self.device,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
        
        # Add cache_dir if specified
        if self.cache_dir:
            model_kwargs["cache_dir"] = self.cache_dir
        
        # Add forced aligner if specified
        if self.forced_aligner_model:
            logger.info(f"Loading forced aligner: {self.forced_aligner_model}")
            model_kwargs["forced_aligner"] = self.forced_aligner_model
            model_kwargs["forced_aligner_kwargs"] = {
                "dtype": self.precision,
                "device_map": self.device,
            }
        
        # Load model using official qwen-asr package
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        
        logger.info("ASR model loaded successfully using qwen-asr package")
    
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        return_timestamps: bool = True,
        forced_aligner: Optional[str] = None,
    ) -> ASRResult:
        """Transcribe audio to text.
        
        Args:
            audio: Path to audio file or numpy array of audio samples.
            sample_rate: Sample rate of audio (required if audio is array).
            language: Language code (auto-detected if None).
            return_timestamps: Whether to return word-level timestamps.
            forced_aligner: Model path for forced aligner to get timestamps.
        
        Returns:
            ASRResult with transcribed text and optional timestamps.
        """
        self._load_model()
        
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio_path = str(audio)
        else:
            # If numpy array, save to temp file or pass as tuple
            audio_path = (audio, sample_rate)
        
        # Prepare transcribe kwargs
        transcribe_kwargs = {
            "audio": audio_path,
            "language": language,
        }
        
        # Add forced aligner for timestamps if requested
        if return_timestamps and forced_aligner:
            transcribe_kwargs["return_time_stamps"] = True
            transcribe_kwargs["forced_aligner"] = forced_aligner
            transcribe_kwargs["forced_aligner_kwargs"] = {
                "dtype": self.precision,
                "device_map": self.device,
            }
        
        # Generate transcription using official qwen-asr API
        results = self._model.transcribe(**transcribe_kwargs)
        result = results[0]
        
        # Extract timestamps if available
        timestamps = []
        if hasattr(result, 'time_stamps') and result.time_stamps:
            timestamps = list(result.time_stamps)
        
        # Calculate audio duration
        try:
            import librosa
            audio_data, sr = librosa.load(audio_path if isinstance(audio_path, str) else audio_path[0], sr=None, mono=True)
            duration = len(audio_data) / sr
        except Exception:
            duration = 0.0
        
        return ASRResult(
            text=result.text,
            language=result.language or "auto",
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
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ASR model unloaded")
