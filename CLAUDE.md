# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an R-based research project for LLM-assisted factor analysis. The pipeline performs exploratory factor analysis (EFA) on psychometric survey data, then uses large language models (LLMs) to generate semantic labels for the discovered factors.

## Architecture

The workflow consists of a single pipeline script (`LLM_Factor_Analysis.R`) with the following sequential stages:

1. **Data Preparation**: Expects two user-provided dataframes in the environment:
   - `question_df_clean`: columns `question_id`, `item_text` (the survey questions)
   - `scale_data_clean`: columns `Q1`, `Q2`, ..., `Qk` (numeric survey responses)

2. **Factor Determination**: Uses parallel analysis (`psych::fa.parallel`) to suggest optimal number of factors

3. **Exploratory Factor Analysis**: Runs maximum likelihood EFA with oblimin rotation via `psych::fa()`

4. **Optional Autoencoder Branch**: Alternative nonlinear dimensionality reduction using Keras (set `use_autoencoder <- TRUE`)

5. **Prompt Construction**: Builds LLM prompts for each factor using:
   - Top loading items (text and loadings)
   - Variance explained statistics
   - Template requesting 1-3 word construct names

6. **LLM Labeling**:
   - Calls OpenAI API (currently uses deprecated `text-davinci-003` completions endpoint, but script references "gpt-5-mini" in comments)
   - Generates base labels and leave-one-out (LOO) variants for stability assessment
   - Extracts labels and joint log-probabilities for ranking

7. **Post-processing**: Ranks candidates, computes stability metrics, exports results

## Key Dependencies

```r
library(psych)      # Factor analysis
library(tidyverse)  # Data manipulation
library(httr)       # API calls
library(jsonlite)   # JSON parsing
library(glue)       # String formatting
library(keras)      # Optional: autoencoder (requires install_keras())
```

## Running the Pipeline

The script is designed to be sourced after loading data into the environment:

```r
# 1. Load your data (ensure question_df_clean and scale_data_clean exist)
# 2. Set API key
Sys.setenv(OPENAI_API_KEY = "your-key-here")

# 3. Source the script (runs entire pipeline automatically)
source("LLM_Factor_Analysis.R")
```

The script will automatically:
- Perform parallel analysis to determine number of factors
- Run EFA and extract defining items
- Generate LLM labels for each factor (calls GPT-4 Turbo API)
- Run leave-one-out stability analysis
- Print results including stability proportions

## Important Notes

- **API Configuration**: Uses OpenAI's `gpt-4-turbo` model via chat completions endpoint. Requires `OPENAI_API_KEY` environment variable.

- **Data Format**: Items must be named `Q1`, `Q2`, etc. in `scale_data_clean`, matching `question_id` values in `question_df_clean`.

- **API Costs**: The script automatically runs multiple API calls (base label + 5 LOO variants per factor). With 2 factors and 5 items each, expect ~12 API calls total.

- **Debug Output**: LOO prompts are printed to console via `print(reduced_prompt)` for verification.
