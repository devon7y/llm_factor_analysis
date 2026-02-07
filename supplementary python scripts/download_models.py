#!/usr/bin/env python3
"""
Script to properly download HuggingFace models for offline use.
Run this while connected to the internet, then use the models offline.
"""

import os

# Set cache directory
os.environ['HF_HOME'] = '/home/devon7y/links/scratch/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/devon7y/links/scratch/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/home/devon7y/links/scratch/huggingface'

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# Models to download
MODELS = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
]

print("="*70)
print("Downloading HuggingFace Models for Offline Use")
print("="*70)
print(f"Cache directory: {os.environ['HF_HOME']}")
print()

for model_name in MODELS:
    print(f"\n{'='*70}")
    print(f"Downloading: {model_name}")
    print(f"{'='*70}")

    try:
        # Use snapshot_download to properly download all files
        cache_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=os.environ['HF_HOME'],
            resume_download=True,  # Resume if partially downloaded
            local_files_only=False,  # Allow internet access
        )
        print(f"✓ Successfully downloaded to: {cache_dir}")

    except Exception as e:
        print(f"✗ Error downloading {model_name}:")
        print(f"  {type(e).__name__}: {str(e)}")
        continue

print(f"\n{'='*70}")
print("Download Complete!")
print(f"{'='*70}")
print()
print("Models are now cached and ready for offline use.")
print(f"Cache location: {os.environ['HF_HOME']}/hub/")
print()
print("To use offline, set local_files_only=True when loading models.")
