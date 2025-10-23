#!/usr/bin/env python3
"""
Generate and Save Word Embeddings using Qwen3-Embedding Models

This script generates embeddings for a word list and saves them to disk for reuse.
This avoids regenerating the same embeddings repeatedly, saving time and compute.

Output format: NumPy .npz files (compressed, efficient)
Multi-GPU support: Automatically uses multiple GPUs if available
"""

# ============================================================================
# CONFIGURATION - Edit these settings
# ============================================================================

# Model Selection - Comment out models you don't want to use
MODEL_NAMES = [
    "Qwen/Qwen3-Embedding-0.6B",     # ~2.4 GB, 1024D embeddings
    # "Qwen/Qwen3-Embedding-4B",     # ~16 GB, 4096D embeddings
    # "Qwen/Qwen3-Embedding-8B",     # ~32 GB, 8192D embeddings
]

# Word list path (hard-coded)
WORD_LIST_PATH = "word_lists/wordfreq_top60000_with_constructs.csv"

# Output directory for saved embeddings
OUTPUT_DIR = "embeddings"

# Processing configuration
BATCH_SIZE = 32              # Batch size per GPU (increase if you have more VRAM)
NORMALIZE_EMBEDDINGS = True  # Normalize embeddings for cosine similarity

# ============================================================================

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # ============================================================================
    # Device Detection with Multi-GPU Support
    # ============================================================================

    print("=" * 70)
    print("WORD EMBEDDING GENERATOR - Qwen3-Embedding Models")
    print("=" * 70)
    print(f"\nDetecting available device(s)...")

    # Check for CUDA GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device('cuda')
        print(f"✓ Found {num_gpus} CUDA GPU(s):")
        for i in range(num_gpus):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

        if num_gpus > 1:
            print(f"\n✓ Multi-GPU mode enabled - will use {num_gpus} GPUs for parallel processing")
            use_multi_gpu = True
        else:
            print(f"\n✓ Using single GPU: {torch.cuda.get_device_name(0)}")
            use_multi_gpu = False

    # Check for Apple MPS
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        num_gpus = 1
        use_multi_gpu = False
        print("✓ Using Apple MPS GPU (Metal Performance Shaders)")

    # Fallback to CPU
    else:
        device = torch.device('cpu')
        num_gpus = 0
        use_multi_gpu = False
        print("✓ Using CPU")

    # Check available memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\nSystem Memory:")
        print(f"  Total: {mem.total / (1024**3):.1f} GB")
        print(f"  Available: {mem.available / (1024**3):.1f} GB")
        print(f"  Used: {mem.percent}%")
    except ImportError:
        print("\n  (psutil not installed - skipping memory check)")

    # ============================================================================
    # Load Word List
    # ============================================================================

    print(f"\n{'='*70}")
    print("Loading word list...")
    print(f"{'='*70}")

    try:
        words_df = pd.read_csv(WORD_LIST_PATH, header=None, names=['word'])
        words = words_df['word'].tolist()

        print(f"✓ Loaded {len(words):,} words from: {WORD_LIST_PATH}")
        print(f"  First 5 words: {words[:5]}")
        print(f"  Last 5 words: {words[-5:]}")

        # Extract word list name for output filename
        word_list_name = Path(WORD_LIST_PATH).stem

    except FileNotFoundError:
        print(f"\n✗ ERROR: Word list not found at: {WORD_LIST_PATH}")
        print("  Please check the path and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR loading word list:")
        print(f"  {type(e).__name__}: {str(e)}")
        sys.exit(1)

    # ============================================================================
    # Create Output Directory
    # ============================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n✓ Output directory: {OUTPUT_DIR}/")

    # ============================================================================
    # Load Models and Create Multi-GPU Pools
    # ============================================================================

    print(f"\n{'='*70}")
    print(f"Loading {len(MODEL_NAMES)} model(s)...")
    print(f"{'='*70}")
    print(f"Selected models: {[m.split('/')[-1] for m in MODEL_NAMES]}")

    all_models = {}
    all_pools = {}

    for model_name in MODEL_NAMES:
        model_size = model_name.split("-")[-1]

        print(f"\n{'='*70}")
        print(f"Loading {model_name}")
        print(f"{'='*70}")
        print(f"This may take 30-120 seconds depending on model size...")

        try:
            # Load model
            model = SentenceTransformer(model_name)

            print(f"✓ {model_size} model loaded successfully!")
            print(f"  Device: {model.device}")
            print(f"  Max sequence length: {model.max_seq_length}")

            # Check embedding dimension
            test_embedding = model.encode(["test"], convert_to_numpy=True)
            embedding_dim = test_embedding.shape[1]
            print(f"  Embedding dimension: {embedding_dim}D")

            # Set up multi-GPU processing if available
            pool = None
            if use_multi_gpu and torch.cuda.is_available() and num_gpus > 1:
                try:
                    print(f"\n  Setting up multi-GPU processing pool...")
                    print(f"  Distributing model across {num_gpus} GPUs...")

                    # Start multi-process pool for parallel GPU processing
                    pool = model.start_multi_process_pool()

                    print(f"  ✓ Multi-GPU pool created successfully!")
                    print(f"    Will use {num_gpus} GPUs for parallel batch processing")

                except Exception as pool_error:
                    print(f"  ⚠ Warning: Could not create multi-GPU pool:")
                    print(f"    {type(pool_error).__name__}: {str(pool_error)}")
                    print(f"    Falling back to single-GPU mode")
                    pool = None

            # Store the model and pool
            all_models[model_size] = model
            all_pools[model_size] = pool

        except Exception as e:
            print(f"\n✗ Error loading {model_name}:")
            print(f"  {type(e).__name__}: {str(e)}")
            print(f"  Skipping this model and continuing with others...")
            continue

    print(f"\n{'='*70}")
    print(f"✓ Successfully loaded {len(all_models)} model(s)!")
    if any(all_pools.values()):
        num_multi_gpu_models = sum(1 for p in all_pools.values() if p is not None)
        print(f"✓ {num_multi_gpu_models} model(s) configured for multi-GPU processing")
    print(f"{'='*70}")

    if len(all_models) == 0:
        print("\n✗ ERROR: No models loaded successfully. Exiting.")
        sys.exit(1)

    # ============================================================================
    # Generate and Save Embeddings
    # ============================================================================

    print(f"\n{'='*70}")
    print(f"Generating embeddings for {len(words):,} words...")
    print(f"{'='*70}")

    saved_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_size, model in sorted(all_models.items()):
        print(f"\n{'='*70}")
        print(f"{model_size} Model - Generating Embeddings")
        print(f"{'='*70}")

        # Check if multi-GPU pool is available for this model
        pool = all_pools.get(model_size, None)

        try:
            # Generate embeddings
            if pool is not None:
                # Multi-GPU processing
                print(f"Using multi-GPU processing across {num_gpus} GPUs...")
                print(f"Batch size: {BATCH_SIZE} per GPU")
                print("This may take 1-3 minutes depending on number of GPUs...")

                embeddings = model.encode_multi_process(
                    words,
                    pool,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=NORMALIZE_EMBEDDINGS
                )
            else:
                # Single GPU/MPS/CPU processing
                print(f"Using single-device processing on {device}...")
                print(f"Batch size: {BATCH_SIZE}")
                print("This may take 2-5 minutes depending on model size and device...")

                embeddings = model.encode(
                    words,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=NORMALIZE_EMBEDDINGS
                )

            print(f"\n✓ Embedding generation complete!")
            print(f"  Shape: {embeddings.shape}")
            print(f"  ({embeddings.shape[0]:,} words × {embeddings.shape[1]} dimensions)")

            # Prepare metadata
            metadata = {
                'model_name': f"Qwen/Qwen3-Embedding-{model_size}",
                'model_size': model_size,
                'embedding_dim': embeddings.shape[1],
                'num_words': len(words),
                'word_list_path': WORD_LIST_PATH,
                'word_list_name': word_list_name,
                'normalized': NORMALIZE_EMBEDDINGS,
                'timestamp': timestamp,
                'device': str(device),
                'multi_gpu': pool is not None,
                'num_gpus': num_gpus if pool is not None else 1
            }

            # Save to .npz file
            output_filename = f"{word_list_name}_{model_size}.npz"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            print(f"\n  Saving embeddings to: {output_path}")

            np.savez_compressed(
                output_path,
                embeddings=embeddings,
                words=np.array(words),
                metadata=np.array(metadata, dtype=object)
            )

            # Verify file was created and check size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✓ File saved successfully!")
            print(f"    Size: {file_size_mb:.1f} MB")

            saved_files.append(output_path)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n✗ Out of memory error with {model_size} model!")
                print(f"  Try reducing BATCH_SIZE or using a smaller model")
                continue
            else:
                raise
        except Exception as e:
            print(f"\n✗ Error generating embeddings for {model_size}:")
            print(f"  {type(e).__name__}: {str(e)}")
            continue

    # ============================================================================
    # Cleanup Multi-GPU Resources
    # ============================================================================

    print(f"\n{'='*70}")
    print("Cleaning up multi-GPU resources...")
    print(f"{'='*70}")

    pools_to_close = [(size, pool) for size, pool in all_pools.items() if pool is not None]

    if pools_to_close:
        print(f"\nClosing {len(pools_to_close)} multi-GPU pool(s)...")

        for model_size, pool in pools_to_close:
            try:
                model = all_models[model_size]
                model.stop_multi_process_pool(pool)
                print(f"  ✓ Closed pool for {model_size} model")
            except Exception as e:
                print(f"  ⚠ Warning: Error closing pool for {model_size}:")
                print(f"    {type(e).__name__}: {str(e)}")
    else:
        print("\nNo multi-GPU pools to close (single GPU/MPS/CPU mode)")

    # ============================================================================
    # Summary
    # ============================================================================

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if saved_files:
        print(f"\n✓ Successfully generated and saved {len(saved_files)} embedding file(s):")
        for filepath in saved_files:
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  • {filepath} ({file_size_mb:.1f} MB)")

        print(f"\n{'='*70}")
        print("How to load these embeddings:")
        print(f"{'='*70}")
        print(f"\nimport numpy as np")
        print(f"\n# Load embeddings")
        print(f"data = np.load('{saved_files[0]}')")
        print(f"embeddings = data['embeddings']  # Shape: (num_words, embedding_dim)")
        print(f"words = data['words']             # Shape: (num_words,)")
        print(f"metadata = data['metadata'].item() # Dictionary with info")
        print(f"\n# Example: Get embedding for a specific word")
        print(f"word_idx = list(words).index('happiness')")
        print(f"word_embedding = embeddings[word_idx]")

        print(f"\n{'='*70}")
        print("✓ All done!")
        print(f"{'='*70}\n")
    else:
        print(f"\n✗ No embedding files were saved.")
        print("  Please check the error messages above.")
        sys.exit(1)
