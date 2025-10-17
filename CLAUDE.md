# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for LLM-based factor analysis on personality psychology constructs, specifically analyzing NEO personality inventory items using transformer embeddings.

## Core Architecture

The codebase demonstrates two equivalent approaches to extracting embeddings from personality test items:

1. **[personality_using_pipeline.py](personality_using_pipeline.py)** - Uses HuggingFace's high-level `pipeline` API
2. **[personality.py](personality.py)** - Uses low-level `AutoModel` and `AutoTokenizer` APIs with HuggingFace `Dataset` objects

Both scripts follow the same workflow:
- Load NEO personality items from `NEO_items.csv` (expected columns: `item`, `construct`)
- Extract [CLS] token embeddings using DistilBERT (`distilbert-base-uncased`)
- Compute cosine similarities between item embeddings
- Compare predicted similarities with observed correlations from `item_corrs.csv`
- Report correlation metrics between predicted and observed values

The scripts are originally Google Colab notebooks converted to Python scripts and include Colab-specific setup code (Google Drive mounting, conditional package installation).

## Running Scripts

### Environment Setup

Install required packages:
```bash
pip install datasets transformers torch pandas scikit-learn
```

### Device Support

Scripts automatically detect and use available accelerators:
- NVIDIA GPUs (CUDA)
- Apple Silicon GPUs (MPS)
- CPU fallback

### Execution

Run either script directly:
```bash
python personality_using_pipeline.py
# or
python personality.py
```

Both require these data files in the working directory:
- `NEO_items.csv` - Contains personality test items
- `item_corrs.csv` - Contains observed item correlations

## Data Files

The repository references but does not include CSV data files. Scripts expect:
- **NEO_items.csv**: Columns `item` (text of personality question), `construct` (personality dimension)
- **item_corrs.csv**: Columns `text_i`, `text_j` (item pairs), `cor` (correlation value)

## Implementation Notes

When modifying or extending these scripts:

- The pipeline version is simpler but less flexible for custom preprocessing
- The manual version provides finer control over tokenization and model inputs
- Both use batch processing (batch_size=8) for efficiency
- Embeddings are extracted from the [CLS] token (index [0, 0] in output tensors)
- The `lower_triangle_flat()` function extracts unique pairwise comparisons excluding diagonal
