#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Installing base dependencies from requirements.txt"
python -m pip install -r requirements.txt

echo "[2/4] Installing qwen-asr without dependency resolution"
python -m pip install --no-deps qwen-asr==0.0.6

echo "[3/4] Installing project in editable mode without dependency resolution"
python -m pip install -e . --no-deps

echo "[4/4] Verifying runtime imports"
python - <<'PY'
import qwen_asr, qwen_tts, transformers
print("qwen_asr:", getattr(qwen_asr, "__version__", "unknown"))
print("qwen_tts:", getattr(qwen_tts, "__version__", "unknown"))
print("transformers:", transformers.__version__)
PY

echo "Installation completed."
