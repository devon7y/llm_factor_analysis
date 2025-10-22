# %% [markdown]
# # Export Common English Word Lists to CSV
# 
# This script exports:
# - Top 60,000 words from `wordfreq`
# - All words from NLTK's English corpus

# %%
import pandas as pd
import os

# %% [markdown]
# ## 1. Load Word Lists

# %%
# --- wordfreq list ---
try:
    from wordfreq import top_n_list
    wordfreq_words = top_n_list("en", n_top=60000)
    print(f"Loaded {len(wordfreq_words):,} words from wordfreq")
except ImportError:
    raise ImportError("Please install wordfreq: pip install wordfreq")

# --- nltk list ---
try:
    import nltk
    nltk.download('words', quiet=True)
    from nltk.corpus import words
    nltk_words = sorted(set(w.lower() for w in words.words()))
    print(f"Loaded {len(nltk_words):,} words from NLTK corpus")
except ImportError:
    raise ImportError("Please install nltk: pip install nltk")

# %% [markdown]
# ## 2. Save to CSV files

# %%
# Create output directory
out_dir = "word_lists"
os.makedirs(out_dir, exist_ok=True)

# Save wordfreq words
pd.DataFrame({"word": wordfreq_words}).to_csv(
    os.path.join(out_dir, "wordfreq_top60000.csv"),
    index=False,
    encoding="utf-8"
)
print("✓ Saved wordfreq_top60000.csv")

# Save nltk words
pd.DataFrame({"word": nltk_words}).to_csv(
    os.path.join(out_dir, "nltk_words.csv"),
    index=False,
    encoding="utf-8"
)
print("✓ Saved nltk_words.csv")

# %% [markdown]
# ## 3. Summary

# %%
print(f"\n{'='*70}")
print(f"✓ Export complete!")
print(f"Files saved to: {os.path.abspath(out_dir)}")
print(f"{'='*70}")