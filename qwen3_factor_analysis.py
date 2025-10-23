# %% [markdown]
# # LLM Factor Analysis - Personality Items (Qwen3-Embedding Models)
# 
# Extracts embeddings from NEO personality items using **Qwen3-Embedding models** and compares predicted similarities with observed correlations.
# 
# **Available Models:**
# - Qwen3-Embedding-0.6B (600M parameters, ~2.4 GB, 1024D)
# - Qwen3-Embedding-4B (4B parameters, ~16 GB, 4096D) 
# - Qwen3-Embedding-8B (8B parameters, ~32 GB, 8192D)
# 
# **Model Configuration:**
# - Precision: FP32 (full precision)
# - Library: sentence-transformers (simplified API)
# - Supports: 100+ languages, MTEB top-ranked performance
# 
# **Getting Started:**
# - **All user settings** are in the **Configuration** section below (model selection, pre-generated embeddings, custom words)
# - Easily enable/disable models by commenting/uncommenting lines in `model_names`
# - Use pre-generated embeddings for faster runs (see Configuration section)

# %% [markdown]
# ## Import Dependencies
# 
# This notebook requires:
# - `sentence-transformers>=2.7.0`
# - `transformers>=4.51.0`
# - `torch>=2.0.0`

# %%
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np
print("  - pandas and numpy loaded")

import torch
print(f"  - torch {torch.__version__} loaded")

# Check if sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    print(f"  - sentence-transformers {sentence_transformers.__version__} loaded")
except ImportError:
    print("\nERROR: sentence-transformers not found!")
    print("Please install: pip install sentence-transformers>=2.7.0")
    raise

# Check transformers version
import transformers
print(f"  - transformers {transformers.__version__} loaded")

# %% [markdown]
# ## Configuration - User Settings
# 
# **Edit these variables to customize your analysis:**
# 
# All user-changeable settings are collected here for easy access.

# %%
# ============================================================================
# MODEL SELECTION - Comment out any models you don't want to run
# ============================================================================
# To disable a model, add a '#' at the start of its line
# To enable a model, remove the '#' from the start of its line
# ============================================================================

model_names = [
    "Qwen/Qwen3-Embedding-0.6B",     # 600M parameters, ~2.4 GB, 1024D
    #"Qwen/Qwen3-Embedding-4B",      # 4B parameters, ~16 GB, 4096D
    #"Qwen/Qwen3-Embedding-8B"       # 8B parameters, ~32 GB, 8192D
]

# ============================================================================
# PRE-GENERATED EMBEDDINGS - Optional paths to pre-generated embedding files
# ============================================================================
# Specify paths for each model size. Use empty string ("") or None to generate on-the-fly.
# Run 'python generate_word_embeddings.py' first to create these files.
# Default location: embeddings/wordfreq_top60000_with_constructs_{model_size}.npz
# ============================================================================

PREGENERATED_EMBEDDINGS = {
    "0.6B": "embeddings/wordfreq_top60000_with_constructs_0.6B.npz",
    #"4B": "embeddings/wordfreq_top60000_with_constructs_4B.npz",
    #"8B": "embeddings/wordfreq_top60000_with_constructs_8B.npz",
}

# ============================================================================
# CUSTOM WORDS - For testing specific words against factor centroids
# ============================================================================
# Option 1: Manually specify words (uncomment and add words below)
custom_words = [
    # Add your words here, one per line
    # Example: "happiness", "sadness", "anger"
]

# Option 2: Load from constructs.csv (uncomment to use)
# import pandas as pd
# constructs_df = pd.read_csv("word_lists/constructs.csv")
# custom_words = constructs_df['word'].tolist()

# Option 3: Select specific constructs from the file (uncomment to use)
# import pandas as pd
# constructs_df = pd.read_csv("word_lists/constructs.csv")
# custom_words = constructs_df['word'].tolist()[:50]  # First 50 constructs

# ============================================================================

print("Configuration loaded successfully!")
print("=" * 70)
print(f"Selected models: {[m.split('/')[-1] for m in model_names]}")
print(f"Custom words specified: {len(custom_words)}")
print("=" * 70)

# %% [markdown]
# ## Load and Process Data

# %%
print("Loading scale...")
scale = pd.read_csv('scales/DASS_items.csv', usecols=['code', 'item', 'factor'])
print(f"Loaded {len(scale)} items")

# Preview the data
scale.head()

# %%
# Extract codes, items and factors for easier access
codes = scale['code'].tolist()
items = scale['item'].tolist()
factors = scale['factor'].tolist()

print(f"Total items: {len(items)}")
print(f"Unique factors: {sorted(set(factors))}")
print(f"Sample codes: {codes[:5]}")

# %% [markdown]
# ## Device Detection and Memory Check

# %%
# Detect available device(s)
print("Detecting available device(s)...")

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
    print("\n  psutil not installed - skipping memory check")
    print("Install with: pip install psutil")

# %% [markdown]
# ## Load Qwen3-Embedding Models
# 
# **Memory Requirements (FP32 full precision):**
# - Qwen3-Embedding-0.6B: ~2.4 GB, 1024D embeddings
# - Qwen3-Embedding-4B: ~16 GB, 4096D embeddings
# - Qwen3-Embedding-8B: ~32 GB, 8192D embeddings
# 
# **Model Selection:** Edit the `model_names` list in the **Configuration** cell at the top of this notebook to choose which models to load.

# %%
# Dictionary to store results
all_embeddings = {}
all_models = {}
all_pools = {}  # Store multi-GPU pools

print(f"Loading {len(model_names)} Qwen3-Embedding model(s)...")
print("=" * 70)
print(f"Selected models: {[m.split('/')[-1] for m in model_names]}")
print("=" * 70)

for model_name in model_names:
    model_size = model_name.split("-")[-1]
    print(f"\n{'='*70}")
    print(f"Loading {model_name}")
    print(f"{'='*70}")
    print(f"This may take 30-120 seconds depending on model size...")
    
    try:
        # Load model - let sentence-transformers handle device placement automatically
        model = SentenceTransformer(model_name)
        
        print(f"✓ {model_size} model loaded successfully!")
        print(f"  Device: {model.device}")
        print(f"  Max sequence length: {model.max_seq_length}")
        
        # Check embedding dimension
        test_embedding = model.encode(["test"], convert_to_numpy=True)
        print(f"  Embedding dimension: {test_embedding.shape[1]}")
        
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
        print(f"\nError loading {model_name}:")
        print(f"  {type(e).__name__}: {str(e)}")
        print(f"  Skipping this model and continuing with others...")
        continue

print(f"\n{'='*70}")
print(f"✓ Successfully loaded {len(all_models)} model(s)!")
if any(all_pools.values()):
    num_multi_gpu = sum(1 for p in all_pools.values() if p is not None)
    print(f"✓ {num_multi_gpu} model(s) configured for multi-GPU processing")
print(f"{'='*70}")

# %% [markdown]
# ## Extract Embeddings for All Models
# 
# Using sentence-transformers' `encode()` method for each model.
# 
# **Processing:**
# - Extract embeddings for all personality items with each model
# - Using `batch_size=8` for efficient processing
# - Results stored in `all_embeddings` dictionary keyed by model size

# %%
print(f"Extracting embeddings for {len(items)} personality items using all models...")
print("=" * 70)

for model_size, model in all_models.items():
    print(f"\n{'='*70}")
    print(f"Processing with {model_size} model")
    print(f"{'='*70}")
    
    # Check if multi-GPU pool is available for this model
    pool = all_pools.get(model_size, None)
    
    try:
        if pool is not None:
            # Multi-GPU processing
            print(f"Using multi-GPU processing across {num_gpus} GPUs...")
            embeddings = model.encode_multi_process(
                items,
                pool,
                batch_size=8,  # Batch size per GPU
                normalize_embeddings=False  # Keep raw embeddings for analysis
            )
        else:
            # Single GPU/MPS/CPU processing
            print(f"Using single-device processing...")
            embeddings = model.encode(
                items,
                batch_size=8,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False  # Keep raw embeddings for analysis
            )
        
        # Store results
        all_embeddings[model_size] = embeddings
        
        print(f"\n✓ Embedding extraction complete for {model_size}!")
        print(f"  Shape: {embeddings.shape}")
        print(f"  ({embeddings.shape[0]} items × {embeddings.shape[1]} dimensions)")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nOut of memory error with {model_size}!")
            print(f"  Try reducing batch_size or using a smaller model")
            raise
        else:
            raise

print(f"\n{'='*70}")
print(f"✓ All embeddings extracted successfully!")
print(f"{'='*70}")
print(f"\nEmbedding dimensions by model:")
for model_size, embeddings in all_embeddings.items():
    print(f"  {model_size}: {embeddings.shape[1]}D")

# %% [markdown]
# ## Inspect Embedding Results

# %%
# Compare embedding dimensions across models
print("Embedding dimensions by model:")
print("=" * 70)
for model_size, embeddings in all_embeddings.items():
    print(f"\n{model_size}:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]}D")
    print(f"  First embedding (first 10 values): {embeddings[0][:10]}")

# %%
# Summary statistics for all models
print("Embedding statistics by model:")
print("=" * 70)
for model_size, embeddings in all_embeddings.items():
    print(f"\n{model_size}:")
    print(f"  Min value: {embeddings.min():.4f}")
    print(f"  Max value: {embeddings.max():.4f}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")

# %%
# Check a specific item across all models
sample_idx = 0
print(f"Sample item #{sample_idx}:")
print(f"  Factor: {factors[sample_idx]}")
print(f"  Text: {items[sample_idx]}")
print("\nEmbedding properties by model:")
print("=" * 70)

for model_size, embeddings in all_embeddings.items():
    print(f"\n{model_size}:")
    print(f"  Embedding shape: {embeddings[sample_idx].shape}")
    print(f"  Embedding norm: {np.linalg.norm(embeddings[sample_idx]):.4f}")

# %% [markdown]
# ## T-SNE Visualization - All Models
# 
# Visualize the high-dimensional embeddings in 2D space using T-SNE, color-coded by personality factor.
# 
# We'll create separate T-SNE plots for each model to compare how different model sizes capture semantic relationships.

# %%
# Import visualization libraries
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

print("Visualization libraries loaded")

# %%
# Prepare data for T-SNE (same across all models)
print("Preparing data for T-SNE...")
print(f"Number of items: {len(factors)}")

# Get unique factors for legend
unique_factors = sorted(set(factors))
print(f"Personality factors: {unique_factors}")

# Create a color map for the personality factors
import matplotlib.cm as cm
colors_map = cm.get_cmap('tab10', len(unique_factors))
factor_to_color = {factor: colors_map(i) for i, factor in enumerate(unique_factors)}

# %%
# Run T-SNE and create visualizations for all models
print("Running T-SNE for all models...")
print("=" * 70)

all_tsne_embeddings = {}

for model_size, embeddings in all_embeddings.items():
    print(f"\n{'='*70}")
    print(f"Running T-SNE for {model_size} model...")
    print(f"{'='*70}")
    print(f"Input shape: {embeddings.shape}")
    
    # Run T-SNE dimensionality reduction
    tsne = TSNE(
        n_components=2,      # Reduce to 2D
        perplexity=25,       # Balance local vs global structure
        max_iter=1000,       # Number of iterations
        random_state=42,     # For reproducibility
        verbose=1            # Show progress
    )
    
    # Transform high-D embeddings to 2D
    embeddings_2d = tsne.fit_transform(embeddings)
    all_tsne_embeddings[model_size] = embeddings_2d
    
    print(f"✓ T-SNE complete! 2D embeddings shape: {embeddings_2d.shape}")

print(f"\n{'='*70}")
print(f"✓ T-SNE complete for all {len(all_tsne_embeddings)} models!")
print(f"{'='*70}")

# %%
# Create T-SNE scatter plots for all models
print("Creating visualizations...")
print("=" * 70)

# Create plots directory if it doesn't exist
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)
print(f"Plots will be saved to: {plots_dir}/")

# Generate timestamp for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Determine number of models
num_models = len(all_tsne_embeddings)
print(f"Creating plots for {num_models} model(s)...")

# Check if we have any models to plot
if num_models == 0:
    print("\n⚠ WARNING: No T-SNE embeddings available to plot!")
    print("  This could mean:")
    print("  - No models were successfully loaded")
    print("  - Embedding extraction failed for all models")
    print("  - T-SNE failed for all models")
    print("\nSkipping visualization...")
else:
    # Adjust figure size based on number of models
    # Use dynamic sizing, with 3 columns max for better layout
    fig_width = min(24, 8 * num_models)  # Cap at 24 inches
    fig, axes = plt.subplots(1, min(3, num_models), figsize=(fig_width, 8))
    
    # Handle case of single model (axes is not a list in this case)
    if min(3, num_models) == 1:
        axes = [axes]
    
    for idx, (model_size, embeddings_2d) in enumerate(sorted(all_tsne_embeddings.items())):
        if idx >= 3:  # Only plot first 3 models in this cell
            break
            
        ax = axes[idx]
        
        # Plot each factor with a different color
        for factor in unique_factors:
            # Get indices for this factor
            indices = [i for i, f in enumerate(factors) if f == factor]
            
            # Plot points for this factor
            ax.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                c=[factor_to_color[factor]],
                label=factor,
                alpha=0.6,
                s=80,
                edgecolors='white',
                linewidth=0.5
            )
        
        # Add labels for each point using the 'code' column
        for i in range(len(embeddings_2d)):
            ax.annotate(
                codes[i],  # Use the code as the label
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=7,
                alpha=0.7,
                ha='center',
                va='bottom',
                xytext=(0, 3),  # Offset label slightly above the point
                textcoords='offset points'
            )
        
        # Get embedding dimension for title
        embedding_dim = all_embeddings[model_size].shape[1]
        
        ax.set_xlabel('T-SNE Component 1', fontsize=11)
        ax.set_ylabel('T-SNE Component 2', fontsize=11)
        ax.set_title(
            f'{model_size} Model\n({embedding_dim}D → 2D)',
            fontsize=13,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Only add legend to the rightmost plot
        if idx == min(2, num_models - 1):
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Overall title
    fig.suptitle(
        'T-SNE Visualization of Scale Item Embeddings\n'
        'Qwen3-Embedding Models',
        fontsize=16,
        fontweight='bold',
        y=1.00
    )
    
    plt.tight_layout()
    
    # Save the figure
    model_names_str = "_".join(sorted(list(all_tsne_embeddings.keys())[:3]))
    filename = f"qwen3_tsne_visualization_{model_names_str}_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {filepath}")
    
    # Display the plot
    plt.show()
    
    print("\n✓ Visualization complete!")

# %% [markdown]
# ## Analyze Nearest Neighbors - All Models
# 
# Compare how different model sizes identify semantic neighbors.

# %%
# Analyze nearest neighbors in the ORIGINAL high-dimensional space for all models
print("Finding nearest neighbors in original embedding space (not T-SNE)...")
print("=" * 70)

from sklearn.metrics.pairwise import cosine_similarity

print(f"\nSample item #{sample_idx}:")
print(f"  Factor: {factors[sample_idx]}")
print(f"  Text: {items[sample_idx]}")

for model_size, embeddings in sorted(all_embeddings.items()):
    print(f"\n{'='*70}")
    print(f"{model_size} Model - Original {embeddings.shape[1]}D Space")
    print(f"{'='*70}")
    
    # Compute cosine similarity between sample and all items
    similarities = cosine_similarity([embeddings[sample_idx]], embeddings)[0]
    
    # Find 5 most similar items (excluding itself)
    most_similar_indices = np.argsort(similarities)[::-1][1:6]
    
    print(f"5 Most similar items (by cosine similarity):")
    for rank, idx in enumerate(most_similar_indices, 1):
        print(f"  {rank}. [{factors[idx]}] {items[idx]}")
        print(f"      Similarity: {similarities[idx]:.4f}")

# %% [markdown]
# ## Quantify Factor Separation
# 
# Measure how well the embeddings separate the three DASS factors (Anxiety, Depression, Stress) using cosine similarity analysis.
# 
# **Metrics:**
# - **Within-factor similarity**: Average cosine similarity between items in the same factor
# - **Between-factor similarity**: Average cosine similarity between items in different factors  
# - **Separation ratio**: Within / Between (higher = better separation, >1.0 means factors cluster together)

# %%
print("Computing factor separation metrics...")
print("=" * 70)

# Loop through all models
for model_size, embeddings in sorted(all_embeddings.items()):
    print(f"\n{'='*70}")
    print(f"{model_size} Model - Factor Separation Analysis")
    print(f"{'='*70}")
    
    # Compute full similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Initialize accumulators
    within_factor_sims = {factor: [] for factor in unique_factors}
    between_factor_sims = []
    
    # Compute within-factor and between-factor similarities
    for i in range(len(items)):
        for j in range(i + 1, len(items)):  # Only upper triangle (avoid duplicates)
            similarity = sim_matrix[i, j]
            
            if factors[i] == factors[j]:
                # Same factor - within-factor similarity
                within_factor_sims[factors[i]].append(similarity)
            else:
                # Different factors - between-factor similarity
                between_factor_sims.append(similarity)
    
    # Compute overall metrics
    all_within_sims = []
    for factor_sims in within_factor_sims.values():
        all_within_sims.extend(factor_sims)
    
    within_mean = np.mean(all_within_sims)
    between_mean = np.mean(between_factor_sims)
    separation_ratio = within_mean / between_mean
    
    # Print overall results
    print(f"\nOverall Separation Metrics:")
    print(f"  Within-factor similarity:  {within_mean:.4f}")
    print(f"  Between-factor similarity: {between_mean:.4f}")
    print(f"  Separation ratio:          {separation_ratio:.4f}")
    print(f"    {'(Good separation - factors cluster together!)' if separation_ratio > 1.0 else '(Poor separation - factors overlap)'}")
    
    # Print per-factor breakdown
    print(f"\nPer-Factor Within-Similarity:")
    for factor in unique_factors:
        factor_mean = np.mean(within_factor_sims[factor])
        factor_std = np.std(within_factor_sims[factor])
        n_pairs = len(within_factor_sims[factor])
        print(f"  {factor:12s}: {factor_mean:.4f} ± {factor_std:.4f}  (n={n_pairs} pairs)")
    
    # Compute pairwise between-factor similarities
    print(f"\nBetween-Factor Similarities:")
    factor_pairs = {}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if factors[i] != factors[j]:
                pair = tuple(sorted([factors[i], factors[j]]))
                if pair not in factor_pairs:
                    factor_pairs[pair] = []
                factor_pairs[pair].append(sim_matrix[i, j])
    
    for pair in sorted(factor_pairs.keys()):
        pair_mean = np.mean(factor_pairs[pair])
        pair_std = np.std(factor_pairs[pair])
        n_pairs = len(factor_pairs[pair])
        print(f"  {pair[0]:12s} vs {pair[1]:12s}: {pair_mean:.4f} ± {pair_std:.4f}  (n={n_pairs} pairs)")

print(f"\n{'='*70}")
print("Factor separation analysis complete!")
print(f"{'='*70}")

# %% [markdown]
# ## Calculate Factor Centroids
# 
# Compute the mean embedding (centroid) for each of the three DASS factors. These centroids represent the "average" embedding for each psychological dimension and can be used for further analysis.

# %%
print("Calculating factor centroids...")
print("=" * 70)

# Dictionary to store centroids for all models
all_centroids = {}

# Loop through all models
for model_size, embeddings in sorted(all_embeddings.items()):
    print(f"\n{'='*70}")
    print(f"{model_size} Model - Computing Centroids")
    print(f"{'='*70}")
    
    # Initialize centroid dictionary for this model
    centroids = {}
    
    # Calculate centroid (mean embedding) for each factor
    for factor in unique_factors:
        # Get indices of items belonging to this factor
        factor_indices = [i for i, f in enumerate(factors) if f == factor]
        
        # Get embeddings for this factor
        factor_embeddings = embeddings[factor_indices]
        
        # Compute centroid (mean of all embeddings in this factor)
        centroid = np.mean(factor_embeddings, axis=0)
        
        # Store centroid
        centroids[factor] = centroid
        
        # Print info
        print(f"\n{factor}:")
        print(f"  Number of items: {len(factor_indices)}")
        print(f"  Centroid shape: {centroid.shape}")
        print(f"  Centroid norm: {np.linalg.norm(centroid):.4f}")
    
    # Store centroids for this model
    all_centroids[model_size] = centroids

print(f"\n{'='*70}")
print("Centroid calculation complete!")
print(f"{'='*70}")
print(f"\nCentroids stored in 'all_centroids' dictionary:")
print(f"  Structure: all_centroids[model_size][factor] = centroid_vector")
print(f"  Models: {list(all_centroids.keys())}")
print(f"  Factors per model: {list(all_centroids[list(all_centroids.keys())[0]].keys())}")

# %% [markdown]
# ## Find Nearest Neighbor Words for Factor Centroids
# 
# Using the computed factor centroids, we'll find the most semantically similar words from a large vocabulary. This analysis identifies which common English words are closest to each psychological dimension.
# 
# **Process:**
# 1. Load 60,000 common English words from wordfreq dataset
# 2. Generate embeddings for all words using the same model(s)
# 3. Compute cosine similarity between each factor centroid and all word embeddings
# 4. Display top 10 nearest neighbor words for each factor (Anxiety, Depression, Stress)

# %% [markdown]
# ## Load or Generate Word Embeddings
# 
# **Performance Optimization:** This section can load pre-generated word embeddings from disk (much faster) or generate them on-the-fly.
# 
# To use pre-generated embeddings:
# 1. Run `python generate_word_embeddings.py` once to create embedding files
# 2. The paths are configured in the **Configuration** cell at the top of this notebook
# 3. This notebook will automatically load them when available
# 
# **Benefits:**
# - Saves 2-5 minutes per model run
# - Ensures consistency across multiple analyses
# - Embeddings are instantly available from disk

# %%
# Display pre-generated embedding configuration status
print("Pre-generated embedding configuration:")
print("=" * 70)
for model_size, path in PREGENERATED_EMBEDDINGS.items():
    if path and os.path.exists(path):
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {model_size}: ✓ Found ({file_size_mb:.1f} MB) - {path}")
    elif path:
        print(f"  {model_size}: ✗ Not found - {path}")
    else:
        print(f"  {model_size}: (will generate on-the-fly)")
print("=" * 70)

# %%
# Load word list
print("Loading word list from dataset...")
print("=" * 70)

word_list_path = "word_lists/wordfreq_top60000_with_constructs.csv"
words_df = pd.read_csv(word_list_path, header=None, names=['word'])
words = words_df['word'].tolist()

print(f"✓ Loaded {len(words):,} words")
print(f"Sample words: {words[:10]}")
print(f"Last 5 words: {words[-5:]}")

# Dictionary to store word embeddings for all models
all_word_embeddings = {}

# Generate or load embeddings for all words using each model
print(f"\n{'='*70}")
print("Loading/Generating word embeddings...")
print(f"{'='*70}")

for model_size, model in sorted(all_models.items()):
    print(f"\n{'='*70}")
    print(f"{model_size} Model")
    print(f"{'='*70}")
    
    # Check if we should use a pre-generated embedding file
    pregenerated_path = PREGENERATED_EMBEDDINGS.get(model_size, None)
    use_pregenerated = False
    
    if pregenerated_path and os.path.exists(pregenerated_path):
        try:
            print(f"Loading pre-generated embeddings from: {pregenerated_path}")
            
            # Load the .npz file
            data = np.load(pregenerated_path, allow_pickle=True)
            
            # Extract embeddings and words
            loaded_embeddings = data['embeddings']
            loaded_words = data['words']
            metadata = data['metadata'].item() if 'metadata' in data else {}
            
            # Convert loaded_words to list of strings for comparison
            loaded_words_list = [str(w) for w in loaded_words]
            
            # Verify the word lists match
            if len(loaded_words_list) == len(words) and loaded_words_list == words:
                word_embeddings = loaded_embeddings
                use_pregenerated = True
                
                print(f"✓ Pre-generated embeddings loaded successfully!")
                print(f"  Shape: {word_embeddings.shape}")
                print(f"  ({word_embeddings.shape[0]:,} words × {word_embeddings.shape[1]} dimensions)")
                
                if metadata:
                    print(f"  Normalized: {metadata.get('normalized', 'unknown')}")
                    print(f"  Generated: {metadata.get('timestamp', 'unknown')}")
                    print(f"  Model: {metadata.get('model_name', 'unknown')}")
            else:
                print(f"  ⚠ Warning: Word list mismatch!")
                print(f"    Expected {len(words)} words, file contains {len(loaded_words_list)}")
                if len(loaded_words_list) == len(words):
                    # Same length but different content - find first difference
                    for i, (expected, actual) in enumerate(zip(words, loaded_words_list)):
                        if expected != actual:
                            print(f"    First difference at index {i}: expected '{expected}', got '{actual}'")
                            break
                print(f"    Falling back to generating embeddings...")
                
        except Exception as e:
            print(f"  ⚠ Warning: Could not load pre-generated embeddings:")
            print(f"    {type(e).__name__}: {str(e)}")
            print(f"    Falling back to generating embeddings...")
    
    # Generate embeddings if we didn't use pre-generated ones
    if not use_pregenerated:
        print(f"Encoding {len(words):,} words...")
        
        # Check if multi-GPU pool is available for this model
        pool = all_pools.get(model_size, None)
        
        if pool is not None:
            # Multi-GPU processing
            print(f"Using multi-GPU processing across {num_gpus} GPUs...")
            print("This may take 1-3 minutes depending on number of GPUs...")
            word_embeddings = model.encode_multi_process(
                words,
                pool,
                batch_size=32,  # Larger batch size per GPU for single words
                normalize_embeddings=True  # Normalize for cosine similarity
            )
        else:
            # Single GPU/MPS/CPU processing
            print("Using single-device processing...")
            print("This may take 2-5 minutes depending on model size and device...")
            word_embeddings = model.encode(
                words,
                batch_size=32,  # Larger batch size for single words
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
        
        print(f"✓ Word embeddings generated!")
        print(f"  Shape: {word_embeddings.shape}")
        print(f"  ({word_embeddings.shape[0]:,} words × {word_embeddings.shape[1]} dimensions)")
    
    # Store the embeddings
    all_word_embeddings[model_size] = word_embeddings

print(f"\n{'='*70}")
print("✓ All word embeddings ready!")
print(f"{'='*70}")

# Summary
num_loaded = sum(1 for ms in all_models.keys() 
                 if PREGENERATED_EMBEDDINGS.get(ms) and os.path.exists(PREGENERATED_EMBEDDINGS.get(ms, "")))
num_generated = len(all_models) - num_loaded

if num_loaded > 0:
    print(f"\nSummary: Loaded {num_loaded} pre-generated, generated {num_generated} on-the-fly")

# %%
# Find nearest neighbor words for each factor centroid
print("Finding nearest neighbor words for each factor centroid...")
print("=" * 70)

# Process each model
for model_size in sorted(all_models.keys()):
    print(f"\n{'='*70}")
    print(f"{model_size} Model - Nearest Neighbor Words")
    print(f"{'='*70}")
    
    # Get centroids and word embeddings for this model
    centroids = all_centroids[model_size]
    word_embeddings = all_word_embeddings[model_size]
    
    # Normalize centroids for cosine similarity (if not already normalized)
    normalized_centroids = {}
    for factor, centroid in centroids.items():
        norm = np.linalg.norm(centroid)
        normalized_centroids[factor] = centroid / norm if norm > 0 else centroid
    
    # For each factor, find top 10 nearest neighbor words
    for factor in unique_factors:
        print(f"\n{'-'*70}")
        print(f"Factor: {factor}")
        print(f"{'-'*70}")
        
        # Compute cosine similarity between this factor's centroid and all word embeddings
        centroid = normalized_centroids[factor]
        similarities = cosine_similarity([centroid], word_embeddings)[0]
        
        # Get indices of top 10 most similar words
        top_indices = np.argsort(similarities)[::-1][:10]
        
        # Display results
        print(f"\nTop 10 nearest neighbor words:")
        print(f"{'Rank':<6} {'Word':<20} {'Similarity':<12}")
        print("-" * 40)
        for rank, idx in enumerate(top_indices, 1):
            word = words[idx]
            similarity = similarities[idx]
            print(f"{rank:<6} {word:<20} {similarity:.4f}")

print(f"\n{'='*70}")
print("✓ Nearest neighbor analysis complete!")
print(f"{'='*70}")

# %% [markdown]
# ## Test Factor Label Names Against Their Own Centroids
# 
# Automatically test how well the factor label names (e.g., "Agreeableness", "Neuroticism") match their own factor centroids. This reveals whether the conventional psychological labels align with the semantic content of their constituent items.

# %%
# Automatically use factor names from the scale
factor_names = unique_factors  # Already extracted from the scale CSV

print(f"Testing {len(factor_names)} factor label name(s) against factor centroids...")
print("=" * 70)
print(f"Factor names: {factor_names}\n")

# Generate embeddings for factor names using each model
factor_name_embeddings = {}

for model_size, model in sorted(all_models.items()):
    print(f"{'='*70}")
    print(f"{model_size} Model - Encoding {len(factor_names)} factor names")
    print(f"{'='*70}")
    
    # Encode factor names
    embeddings = model.encode(
        factor_names,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )
    
    factor_name_embeddings[model_size] = embeddings
    print(f"✓ Factor name embeddings generated!")
    print(f"  Shape: {embeddings.shape}\n")

# Compute similarities for each model
for model_size in sorted(all_models.keys()):
    print(f"{'='*70}")
    print(f"{model_size} Model - Factor Name Similarities")
    print(f"{'='*70}")
    
    # Get centroids and factor name embeddings for this model
    centroids = all_centroids[model_size]
    embeddings = factor_name_embeddings[model_size]
    
    # Normalize centroids
    normalized_centroids = {}
    for factor, centroid in centroids.items():
        norm = np.linalg.norm(centroid)
        normalized_centroids[factor] = centroid / norm if norm > 0 else centroid
    
    # Compute similarity matrix: factor names × factor centroids
    similarity_matrix = np.zeros((len(factor_names), len(unique_factors)))
    
    for factor_idx, factor in enumerate(unique_factors):
        centroid = normalized_centroids[factor]
        similarities = cosine_similarity([centroid], embeddings)[0]
        similarity_matrix[:, factor_idx] = similarities
    
    # Create DataFrame for easy viewing
    results_df = pd.DataFrame(
        similarity_matrix,
        columns=unique_factors,
        index=factor_names
    )
    
    # Add a column for the highest similarity factor
    results_df['Highest'] = results_df.idxmax(axis=1)
    results_df['Max_Sim'] = results_df[unique_factors].max(axis=1)
    
    # Sort by maximum similarity (descending)
    results_df = results_df.sort_values('Max_Sim', ascending=False)
    
    # Display results in a clearer format
    print(f"\nSimilarity scores for {len(factor_names)} factor label name(s):")
    print(f"(Sorted by maximum similarity)\n")
    
    # Display each factor name with its similarities
    for factor_name in results_df.index:
        max_factor = results_df.loc[factor_name, 'Highest']
        max_sim = results_df.loc[factor_name, 'Max_Sim']
        
        # Check if the factor name matches its own centroid
        is_correct = (factor_name == max_factor)
        marker = "✓" if is_correct else "✗"
        
        print(f"\n{factor_name}: {marker}")
        print(f"  → {max_factor}: {max_sim:.4f} ★")
        
        # Show other factors
        for factor in unique_factors:
            if factor != max_factor:
                sim = results_df.loc[factor_name, factor]
                # Highlight if this is the "correct" factor
                if factor == factor_name:
                    print(f"  → {factor}: {sim:.4f} (expected match)")
                else:
                    print(f"  → {factor}: {sim:.4f}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics:")
    print(f"{'='*70}")
    
    # Count how many factor names correctly matched their centroids
    correct_matches = sum(results_df.index == results_df['Highest'])
    print(f"Correct matches: {correct_matches}/{len(factor_names)} ({correct_matches/len(factor_names)*100:.1f}%)\n")
    
    for factor in unique_factors:
        mean_sim = results_df[factor].mean()
        max_sim = results_df[factor].max()
        max_word = results_df[factor].idxmax()
        # Check if this factor's name is the best match
        self_match = results_df.loc[factor, factor] if factor in results_df.index else 0.0
        print(f"{factor:20s} → Mean: {mean_sim:.4f}, Max: {max_sim:.4f} ({max_word}), Self: {self_match:.4f}")

print(f"\n{'='*70}")
print("✓ Factor name analysis complete!")
print(f"{'='*70}")

# %% [markdown]
# ## Test Specific Words Against Factor Centroids
# 
# Test custom words against factor centroids to measure their similarity to each psychological dimension. This is useful for:
# - Testing theoretically-relevant psychological constructs
# - Validating expected associations
# - Exploring semantic relationships with specific terms
# 
# **To use this section:** Edit the `custom_words` variable in the Configuration cell at the top of this notebook.

# %%
# Test custom words against factor centroids
# (Custom words are configured in the Configuration cell at the top of the notebook)

# Skip this analysis if no words specified
if not custom_words:
    print("No custom words specified. Skipping this analysis.")
    print("\nTo use this section:")
    print("  1. Edit the 'custom_words' list in the Configuration cell at the top, OR")
    print("  2. Uncomment one of the constructs.csv loading options in the Configuration cell")
else:
    print(f"Testing {len(custom_words)} custom word(s) against factor centroids...")
    print("=" * 70)
    
    # Generate embeddings for custom words using each model
    custom_word_embeddings = {}
    
    for model_size, model in sorted(all_models.items()):
        print(f"\n{'='*70}")
        print(f"{model_size} Model - Encoding {len(custom_words)} custom words")
        print(f"{'='*70}")
        
        # Encode custom words
        embeddings = model.encode(
            custom_words,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        custom_word_embeddings[model_size] = embeddings
        print(f"✓ Custom word embeddings generated!")
        print(f"  Shape: {embeddings.shape}")
    
    # Compute similarities for each model
    for model_size in sorted(all_models.keys()):
        print(f"\n{'='*70}")
        print(f"{model_size} Model - Custom Word Similarities")
        print(f"{'='*70}")
        
        # Get centroids and custom word embeddings for this model
        centroids = all_centroids[model_size]
        embeddings = custom_word_embeddings[model_size]
        
        # Normalize centroids
        normalized_centroids = {}
        for factor, centroid in centroids.items():
            norm = np.linalg.norm(centroid)
            normalized_centroids[factor] = centroid / norm if norm > 0 else centroid
        
        # Compute similarity matrix: words × factors
        similarity_matrix = np.zeros((len(custom_words), len(unique_factors)))
        
        for factor_idx, factor in enumerate(unique_factors):
            centroid = normalized_centroids[factor]
            similarities = cosine_similarity([centroid], embeddings)[0]
            similarity_matrix[:, factor_idx] = similarities
        
        # Create DataFrame for easy viewing
        results_df = pd.DataFrame(
            similarity_matrix,
            columns=unique_factors,
            index=custom_words
        )
        
        # Add a column for the highest similarity factor
        results_df['Highest'] = results_df.idxmax(axis=1)
        results_df['Max_Sim'] = results_df[unique_factors].max(axis=1)
        
        # Sort by maximum similarity (descending)
        results_df = results_df.sort_values('Max_Sim', ascending=False)
        
        # Display results in a clearer format
        print(f"\nSimilarity scores for {len(custom_words)} custom word(s):")
        print(f"(Sorted by maximum similarity)\n")
        
        # Display each word with its similarities
        for word in results_df.index:
            max_factor = results_df.loc[word, 'Highest']
            max_sim = results_df.loc[word, 'Max_Sim']
            
            print(f"\n{word}:")
            print(f"  → {max_factor}: {max_sim:.4f} ★")
            
            # Show other factors
            for factor in unique_factors:
                if factor != max_factor:
                    sim = results_df.loc[word, factor]
                    print(f"  → {factor}: {sim:.4f}")
        
        # Summary statistics
        print(f"\n{'='*70}")
        print("Summary Statistics:")
        print(f"{'='*70}")
        for factor in unique_factors:
            mean_sim = results_df[factor].mean()
            max_sim = results_df[factor].max()
            max_word = results_df[factor].idxmax()
            print(f"{factor:20s} → Mean: {mean_sim:.4f}, Max: {max_sim:.4f} ({max_word})")
    
    print(f"\n{'='*70}")
    print("✓ Custom word analysis complete!")
    print(f"{'='*70}")

# %% [markdown]
# ## Cleanup Multi-GPU Resources
# 
# If you're done with all analyses, run this cell to properly close multi-GPU pools and free GPU memory.

# %%
# Cleanup multi-GPU pools to free resources
print("Cleaning up multi-GPU resources...")
print("=" * 70)

pools_to_close = []
for model_size, pool in all_pools.items():
    if pool is not None:
        pools_to_close.append((model_size, pool))

if pools_to_close:
    print(f"\nClosing {len(pools_to_close)} multi-GPU pool(s)...")
    
    for model_size, pool in pools_to_close:
        try:
            model = all_models[model_size]
            model.stop_multi_process_pool(pool)
            print(f"  ✓ Closed pool for {model_size} model")
            all_pools[model_size] = None
        except Exception as e:
            print(f"  ⚠ Warning: Error closing pool for {model_size}:")
            print(f"    {type(e).__name__}: {str(e)}")
    
    print(f"\n✓ Multi-GPU resources released!")
else:
    print("\nNo multi-GPU pools to close (single GPU/MPS/CPU mode)")

print(f"{'='*70}")
print("✓ Cleanup complete!")


