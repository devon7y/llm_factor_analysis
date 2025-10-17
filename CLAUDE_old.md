# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an R-based research project for LLM-assisted factor analysis. The repository contains two complementary pipelines that perform exploratory factor analysis (EFA) on psychometric survey data and generate semantic labels for discovered factors using different approaches.

## Repository Structure

### 1. `LLM_Factor_Analysis.R` - GPT-4 with LOO Stability Testing

Primary pipeline focused on LLM-based labeling with robustness validation.

**Workflow stages:**

1. **Data Preparation**: Expects two user-provided dataframes in the environment:
   - `question_df_clean`: columns `question_id`, `item_text` (the survey questions)
   - `scale_data_clean`: columns `Q1`, `Q2`, ..., `Qk` (numeric survey responses)

2. **Factor Determination**: Uses parallel analysis (`psych::fa.parallel`) to suggest optimal number of factors

3. **Exploratory Factor Analysis**: Runs maximum likelihood EFA with oblimin rotation via `psych::fa()`

4. **Prompt Construction**: Builds LLM prompts for each factor using:
   - Items with loadings ≥ 0.25 (configurable threshold)
   - Loading magnitudes displayed in prompts for implicit weighting
   - Variance explained statistics

5. **LLM Labeling (GPT-4 Turbo)**:
   - Generates base label using all items for each factor
   - Runs leave-one-out (LOO) stability testing: N additional API calls per factor
   - Each LOO variant removes one item to test label robustness

6. **Stability Metrics**: Computes `stability_prop` = proportion of LOO labels matching base label

**Running:**
```r
Sys.setenv(OPENAI_API_KEY = "your-key-here")
source("LLM_Factor_Analysis.R")
```

**Output:** `labels_df` with base label + LOO variants, `labels_summary` with stability proportions

---

### 2. `Word2vec_Bert_Factor_Analysis.R` - Multi-Method Comparison

Alternative pipeline comparing 5 different labeling approaches (no LLM API calls required for 4/5 methods).

**Labeling Methods:**

1. **Word2Vec (baseline)**: Finds word in GloVe vocabulary most similar to loading-weighted item centroid
2. **Contrastive Word2Vec**: Scores words by `sim(word, this_factor) - avg_sim(word, other_factors)` for discriminative labels
3. **BERT**: Uses sentence transformers to find most representative item, extracts key phrases
4. **TF-IDF**: Identifies words with high term frequency in this factor but low in others (discriminative)
5. **KeyBERT**: State-of-the-art keyword extraction using BERT embeddings

**Workflow stages:**

1. **Data Preparation & EFA**: Same as Script 1
2. **Pre-computation**:
   - TF-IDF across all factors (one document per factor)
   - Word2Vec centroids for contrastive scoring
3. **Labeling**: Runs all 5 methods for each factor
4. **Comparison Table**: Shows labels from each method side-by-side

**Running:**
```r
# One-time setup: download GloVe embeddings
# https://nlp.stanford.edu/projects/glove/
# Place glove.6B.100d.txt in project directory

# Uncomment lines 55-56 to load GloVe:
# glove <- read.delim("glove.6B.100d.txt", quote = "", comment.char = "", header = FALSE, sep = " ")
# word_vectors <- as.matrix(glove[, -1]); rownames(word_vectors) <- glove[, 1]

source("Word2vec_Bert_Factor_Analysis.R")
```

**Output:**
```
factor   word2vec  contrastive_w2v  bert         tfidf              keybert
Factor_1 Postpone  Delay Postpone   Last Minute  Delay Postpone     Procrastinate
Factor_2 Anxious   Guilty Stress    Work Left    Guilty Stress      Task Anxiety
```

## Dependencies

### R Packages (both scripts)
```r
install.packages(c("psych", "tidyverse", "glue"))
```

### Additional for `Word2vec_Bert_Factor_Analysis.R`
```r
install.packages(c("text2vec", "reticulate", "tidytext"))
```

### Python Setup (for Word2vec_Bert script)
```bash
# Create virtual environment (already done)
python3 -m venv .venv

# Install packages (already done)
.venv/bin/pip install sentence-transformers keybert
```

The script automatically uses the `.venv/` virtual environment via `use_virtualenv(".venv")`.

### External Data
- **GloVe embeddings** (for Word2Vec methods): Download `glove.6B.100d.txt` from https://nlp.stanford.edu/projects/glove/
- Place in project root directory
- ~862MB file, not tracked in git (see `.gitignore`)

## Important Notes

### Data Format Requirements
- Items must be named `Q1`, `Q2`, etc. in `scale_data_clean`
- `question_id` values in `question_df_clean` must match item names
- Both dataframes must exist in R environment before sourcing scripts

### API Costs (`LLM_Factor_Analysis.R`)
- Uses OpenAI's `gpt-4-turbo` model
- Requires `OPENAI_API_KEY` environment variable
- **Cost estimate**: With 2 factors and 18 items each = ~38 API calls total
  - Factor 1: 1 base + 18 LOO = 19 calls
  - Factor 2: 1 base + 18 LOO = 19 calls

### Loading Threshold
- Default: `loading_thresh = 0.25` (items with |loading| ≥ 0.25 included)
- Adjust in line 67 (LLM_Factor_Analysis.R) or line 49 (Word2vec_Bert_Factor_Analysis.R)
- Higher threshold = fewer items, faster runtime, potentially more stable labels
- Lower threshold = more items, captures more factor variance

### Method Comparison
| Feature | LLM_Factor_Analysis.R | Word2vec_Bert_Factor_Analysis.R |
|---------|----------------------|----------------------------------|
| **Purpose** | Stability testing | Method comparison |
| **Methods** | 1 (GPT-4) | 5 (Word2Vec, Contrastive, BERT, TF-IDF, KeyBERT) |
| **API Costs** | High (1+N calls per factor) | None (all methods local except optional GPT) |
| **Output** | Stability metrics | Side-by-side label comparison |
| **Best for** | Publication robustness analysis | Exploring different labeling approaches |
