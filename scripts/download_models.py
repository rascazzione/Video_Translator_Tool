#!/usr/bin/env python3
"""
Download Qwen3 models for Video Translator.

This script downloads the required models from HuggingFace Hub
and saves them to the local cache directory.

Usage:
    python scripts/download_models.py [--model MODEL] [--all]

Examples:
    # Download all models
    python scripts/download_models.py --all

    # Download only ASR model
    python scripts/download_models.py --model asr

    # Download only TTS model
    python scripts/download_models.py --model tts
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Models configuration
MODELS = {
    "asr-1.7b": {
        "repo": "Qwen/Qwen3-ASR-1.7B",
        "description": "ASR model (1.7B parameters, high accuracy)",
        "size_estimate": "~4.5 GB",
    },
    "asr-0.6b": {
        "repo": "Qwen/Qwen3-ASR-0.6B",
        "description": "ASR model (0.6B parameters, faster)",
        "size_estimate": "~2.5 GB",
    },
    "tts-custom": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "description": "TTS model with CustomVoice (1.7B)",
        "size_estimate": "~4.5 GB",
    },
    "tts-design": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "description": "TTS model with VoiceDesign (1.7B)",
        "size_estimate": "~4.5 GB",
    },
    "tts-base": {
        "repo": "Qwen/Qwen3-TTS-25Hz-0.6B-Base",
        "description": "TTS base model (0.6B, faster)",
        "size_estimate": "~2.5 GB",
    },
    "aligner": {
        "repo": "Qwen/Qwen3-ForcedAligner-0.6B",
        "description": "Forced Aligner for timestamps",
        "size_estimate": "~1.0 GB",
    },
}

DEFAULT_MODELS = ["asr-1.7b", "tts-custom", "aligner"]


def download_model(
    repo: str,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Download a model from HuggingFace Hub.
    
    Args:
        repo: HuggingFace model repository ID.
        cache_dir: Local cache directory.
        force: Force re-download if already exists.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface-hub not installed. Run: pip install huggingface-hub")
        return False
    
    print(f"📥 Downloading: {repo}")
    
    try:
        snapshot_download(
            repo_id=repo,
            cache_dir=cache_dir,
            force_download=force,
            show_progress=True,
        )
        print(f"✅ Downloaded: {repo}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {repo}: {e}")
        return False


def list_models() -> None:
    """List all available models."""
    print("\n📦 Available Models:\n")
    print(f"{'Model':<15} {'Description':<50} {'Size':<15}")
    print("-" * 80)
    
    for key, model in MODELS.items():
        print(f"{key:<15} {model['description']:<50} {model['size_estimate']:<15}")
    
    print("\n" + "-" * 80)
    print(f"\nDefault models: {', '.join(DEFAULT_MODELS)}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Qwen3 models for Video Translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        action="append",
        help="Model to download (can be specified multiple times)",
    )
    
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Download all available models",
    )
    
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available models",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./models_cache",
        help="Output directory for models (default: ./models_cache)",
    )
    
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download if models already exist",
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        list_models()
        return 0
    
    # Determine which models to download
    if args.all:
        models_to_download = list(MODELS.keys())
    elif args.model:
        models_to_download = args.model
    else:
        # Default models
        models_to_download = DEFAULT_MODELS
    
    # Validate model names
    invalid_models = [m for m in models_to_download if m not in MODELS]
    if invalid_models:
        print(f"❌ Invalid model(s): {', '.join(invalid_models)}")
        print("Run with --list to see available models")
        return 1
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📥 Models to download: {', '.join(models_to_download)}\n")
    
    # Download models
    success_count = 0
    failed_models = []
    
    for model_key in models_to_download:
        model_info = MODELS[model_key]
        
        if download_model(
            repo=model_info["repo"],
            cache_dir=str(output_path),
            force=args.force,
        ):
            success_count += 1
        else:
            failed_models.append(model_key)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Successfully downloaded: {success_count}/{len(models_to_download)}")
    
    if failed_models:
        print(f"❌ Failed: {', '.join(failed_models)}")
        return 1
    
    print("\n🎉 All models downloaded successfully!")
    print("\n💡 Usage:")
    print(f"   Set MODEL_CACHE_DIR={output_path.absolute()} in your .env file")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
