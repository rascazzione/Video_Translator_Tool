# Installation Guide

This guide covers all installation methods for Video Translator.

## Prerequisites

### Required

- **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
- **FFmpeg** - [Installation instructions](https://ffmpeg.org/download.html)

### Recommended

- **NVIDIA GPU** with 8GB+ VRAM (for 0.6B models) or 16GB+ VRAM (for 1.7B models)
- **CUDA 11.8+** - For GPU acceleration
- **16GB+ RAM** - For smooth processing

---

## Quick Start

### 1. Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```powershell
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### 2. Clone and Install

```bash
# Clone repository
git clone https://github.com/yourusername/video_translator.git
cd video_translator

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### 3. Download Models

```bash
# Download default models (ASR 1.7B, TTS 1.7B, Aligner)
python scripts/download_models.py

# Or download all models
python scripts/download_models.py --all
```

### 4. Configure

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration as needed
nano .env
```

### 5. Test Installation

```bash
# Check CLI
video-translator --version

# Check system info
video-translator info
```

---

## Installation Methods

### Method 1: Pip Installation

```bash
pip install video-translator
```

### Method 2: From Source

```bash
git clone https://github.com/yourusername/video_translator.git
cd video_translator
pip install -e .
```

### Method 3: Docker

```bash
# Build image
docker-compose build

# Run container
docker-compose up -d

# Access CLI
docker-compose exec video-translator video-translator --help
```

---

## GPU Acceleration

### NVIDIA CUDA

1. **Install CUDA Toolkit** (11.8 or later)
   - Download from: https://developer.nvidia.com/cuda-downloads

2. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install FlashAttention 2** (optional, for 2-3x speedup):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

### Apple Silicon (M1/M2/M3)

1. **Install PyTorch with MPS:**
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Set device to MPS:**
   ```bash
   export DEVICE=mps
   ```

### CPU Only

For CPU-only operation:

```bash
export DEVICE=cpu
export PRECISION=fp32
```

Note: Processing will be significantly slower on CPU.

---

## Model Selection

### For Limited VRAM (< 8GB)

Use 0.6B models:

```bash
# In .env file
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-25Hz-0.6B-Base
QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B
```

### For Standard GPU (8-16GB VRAM)

Use 1.7B ASR + 0.6B TTS:

```bash
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-25Hz-0.6B-Base
```

### For High-End GPU (16GB+ VRAM)

Use full 1.7B models:

```bash
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B
```

---

## Troubleshooting

### FFmpeg Not Found

**Error:** `RuntimeError: FFmpeg not found`

**Solution:**
```bash
# Verify installation
ffmpeg -version

# If not found, install:
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (add to PATH)
setx PATH "%PATH%;C:\Program Files\ffmpeg\bin"
```

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Use smaller models (0.6B instead of 1.7B)
2. Reduce precision: `PRECISION=fp16` or `PRECISION=fp32`
3. Close other GPU applications
4. Use CPU for one of the models

### FlashAttention Installation Failed

**Error:** `pip install flash-attn` fails

**Solutions:**
1. Ensure CUDA is properly installed
2. Try specific version:
   ```bash
   pip install flash-attn==2.4.0 --no-build-isolation
   ```
3. Skip FlashAttention (will use sdpa instead):
   ```bash
   FLASH_ATTENTION=false
   ```

### Model Download Failed

**Error:** HuggingFace download timeout

**Solutions:**
1. Use mirror:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python scripts/download_models.py
   ```

2. Download manually from HuggingFace and place in `models_cache/`

3. Use resume capability:
   ```bash
   python scripts/download_models.py --force
   ```

---

## Verification

Run these commands to verify your installation:

```bash
# Check Python version
python --version  # Should be 3.10+

# Check FFmpeg
ffmpeg -version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Video Translator
video-translator info
```

---

## Next Steps

After successful installation:

1. **Try a simple transcription:**
   ```bash
   video-translator transcribe video.mp4 -o output/
   ```

2. **Generate speech from text:**
   ```bash
   echo "Hello world" > test.txt
   video-translator tts test.txt -o speech.wav
   ```

3. **Full video translation:**
   ```bash
   video-translator translate-video input.mp4 es -o translated/
   ```

For more usage examples, see the [Usage Guide](usage.md).
