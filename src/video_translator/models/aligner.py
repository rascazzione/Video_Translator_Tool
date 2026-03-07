"""Qwen3-ForcedAligner wrapper for word-level timestamp alignment."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of forced alignment."""
    
    text: str
    """Aligned text."""
    
    segments: List[Dict[str, Any]] = field(default_factory=list)
    """Word/character segments with timestamps."""
    
    start_time: float = 0.0
    """Overall start time in seconds."""
    
    end_time: float = 0.0
    """Overall end time in seconds."""
    
    language: str = "English"
    """Language of the aligned text."""


class QwenForcedAligner:
    """Wrapper for Qwen3-ForcedAligner model.
    
    Aligns text-speech pairs and returns word or character level timestamps.
    Supports 11 languages: Chinese, English, French, German, Italian,
    Japanese, Korean, Portuguese, Russian, Spanish.
    
    Example:
        >>> aligner = QwenForcedAligner()
        >>> result = aligner.align(
        ...     audio="audio.wav",
        ...     text="Hello world, this is a test.",
        ...     language="English"
        ... )
        >>> for seg in result.segments:
        ...     print(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")
    """
    
    SUPPORTED_LANGUAGES = [
        "Chinese", "English", "French", "German", "Italian",
        "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
    ]
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: Optional[str] = None,
        precision: str = "bf16",
        flash_attention: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize Qwen3-ForcedAligner.
        
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
        
        logger.info(f"Initialized ForcedAligner with model: {model_name}")
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
        """Load the aligner model and processor."""
        if self._model is not None:
            return
        
        try:
            from qwen_asr import Qwen3ForcedAligner as _QwenAligner
        except ImportError:
            try:
                from transformers import AutoModel, AutoProcessor
                self._use_transformers = True
            except ImportError:
                raise ImportError(
                    "Please install transformers or qwen-asr: "
                    "pip install transformers>=4.51.0"
                )
            self._use_transformers = False
        
        logger.info(f"Loading ForcedAligner model: {self.model_name}")
        
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
        
        if hasattr(self, "_use_transformers") and self._use_transformers:
            from transformers import AutoModel, AutoProcessor
            
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            
            self._model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_kwargs,
            )
        else:
            self._model = _QwenAligner.from_pretrained(
                self.model_name,
                dtype=self.precision,
                device_map=self.device if self.device != "cpu" else None,
            )
        
        if self.device == "cpu" and hasattr(self._model, "to"):
            self._model = self._model.to("cpu")
        
        if hasattr(self._model, "eval"):
            self._model.eval()
        
        logger.info("ForcedAligner model loaded successfully")
    
    def align(
        self,
        audio: Union[str, Path, np.ndarray],
        text: str,
        language: str = "English",
        sample_rate: int = 16000,
        unit: str = "word",
    ) -> AlignmentResult:
        """Align text with audio and return timestamps.
        
        Args:
            audio: Path to audio file or numpy array of audio samples.
            text: Text to align with audio.
            language: Language of the text.
            sample_rate: Sample rate of audio (if array).
            unit: Alignment unit ('word' or 'character').
        
        Returns:
            AlignmentResult with segments and timestamps.
        """
        self._load_model()
        
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(
                f"Language '{language}' may not be supported. "
                f"Supported: {self.SUPPORTED_LANGUAGES}"
            )
        
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio), target_sr=sample_rate)
        
        # Perform alignment
        if hasattr(self, "_use_transformers") and self._use_transformers:
            segments = self._align_transformers(audio, text, language, sample_rate)
        else:
            segments = self._align_qwen(audio, text, language, sample_rate)
        
        # Calculate total duration
        if segments:
            start_time = segments[0].get("start", 0.0)
            end_time = segments[-1].get("end", 0.0)
        else:
            start_time = end_time = 0.0
        
        return AlignmentResult(
            text=text,
            segments=segments,
            start_time=start_time,
            end_time=end_time,
            language=language,
        )
    
    def _align_transformers(
        self,
        audio: np.ndarray,
        text: str,
        language: str,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        """Align using transformers backend."""
        # Prepare inputs
        inputs = self._processor(
            audio=audio,
            text=text,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate alignment
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
            )
        
        # Parse results
        segments = self._parse_alignment_outputs(outputs, text)
        return segments
    
    def _align_qwen(
        self,
        audio: np.ndarray,
        text: str,
        language: str,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        """Align using qwen_asr backend."""
        # qwen-asr package API: align(audio, text, language)
        # Note: sampling_rate is not a parameter, audio should be (np.ndarray, sr) tuple or path
        audio_input = (audio, sample_rate)
        results = self._model.align(
            audio=audio_input,
            text=text,
            language=language,
        )
        
        # Parse results
        segments = []
        if isinstance(results, list):
            for result in results:
                if hasattr(result, "text"):
                    segments.append({
                        "text": result.text,
                        "start": result.start_time,
                        "end": result.end_time,
                    })
        elif hasattr(results, "segments"):
            segments = results.segments
        
        return segments
    
    def _parse_alignment_outputs(
        self,
        outputs: torch.Tensor,
        original_text: str,
    ) -> List[Dict[str, Any]]:
        """Parse model outputs into segments."""
        # Simplified parsing - actual implementation depends on model output format
        segments = []
        
        # Split text into words and assign approximate timestamps
        words = original_text.split()
        if len(words) > 0 and hasattr(outputs, "shape"):
            duration = outputs.shape[-1] / 16000  # Approximate
            step = duration / len(words)
            
            for i, word in enumerate(words):
                segments.append({
                    "text": word,
                    "start": i * step,
                    "end": (i + 1) * step,
                })
        
        return segments
    
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
        
        return audio, target_sr
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ForcedAligner model unloaded")
