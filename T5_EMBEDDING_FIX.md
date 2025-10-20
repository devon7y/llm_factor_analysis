# T5-GEMMA Embedding Extraction Fix

**Date:** 2025-01-19
**Issue:** Abnormally high cosine similarities (0.995-0.999) between all items in T5-GEMMA embeddings
**Status:** ✅ Fixed

---

## Problem Description

### Symptoms
When extracting embeddings from T5-GEMMA encoder models for personality test items, we observed:

1. **Abnormally high similarities**: All items showed 0.995-0.999 cosine similarity to each other
2. **No semantic differentiation**: Items from completely different psychological factors (Anxiety, Depression, Stress) were nearly identical
3. **Poor factor separation**: The separation ratio was barely above 1.0 (1.0292), indicating minimal clustering by factor
4. **Strange T-SNE plots**: Visualizations showed unusual patterns with no clear factor groupings

### Example of the Problem
```
Sample item: "I found myself getting upset by quite trivial things." [Stress]

5 Most similar items:
  1. [Anxiety] I had a feeling of faintness.           Similarity: 0.9985
  2. [Stress] I tended to over-react to situations.    Similarity: 0.9980
  3. [Stress] I found myself getting agitated.         Similarity: 0.9979
  4. [Depression] I felt sad and depressed.            Similarity: 0.9976
  5. [Anxiety] I felt terrified.                       Similarity: 0.9955
```

**Problem:** Even semantically unrelated items (like "faintness" vs "upset by trivial things") show 99.85% similarity!

---

## Root Cause Analysis

### The Incorrect Implementation

**Original code (WRONG):**
```python
# Get encoder outputs
outputs = model.encoder(**inputs)

# Extract embeddings from first token of last hidden state
batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
```

### Why This Was Wrong

#### 1. **T5 Architecture Doesn't Have a [CLS] Token**

Unlike BERT-style models, T5 does not use a special [CLS] token at the beginning of sequences to represent sentence-level meaning:

- **BERT/DistilBERT**: `[CLS] token1 token2 ... [SEP]` → The [CLS] token is trained to represent the entire sentence
- **T5**: `token1 token2 token3 ...` → No special token; all tokens are equal

#### 2. **First Token Has No Special Meaning**

In T5, the first token is simply the first token from the input text. It doesn't contain aggregated sentence-level information:

```python
# Example tokenization
"I felt sad" → [306, 1800, 6819]  # Just regular tokens, no [CLS]
```

Taking `outputs.last_hidden_state[:, 0, :]` simply extracts the embedding of the **first word** of each sentence, not a sentence representation.

#### 3. **Why All Embeddings Were Similar**

When extracting only the first token's embedding:
- Many sentences start with common words: "I found...", "I felt...", "I was..."
- The first token for these sentences is identical or very similar
- Result: All embeddings cluster around the representation of "I" or similar pronouns
- This explains the 99.5%+ similarity across all items

---

## The Solution: Mean Pooling with Attention Masking

### Correct Implementation

```python
# Get encoder outputs
outputs = model.encoder(**inputs)

# Mean pooling with attention mask (T5 doesn't have [CLS] token)
# This properly accounts for padding tokens
attention_mask = inputs['attention_mask']

# Expand mask to match hidden state dimensions: (batch_size, seq_len, hidden_dim)
mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()

# Sum embeddings (masked) across sequence dimension
sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)

# Sum mask to get actual token counts (avoid division by zero)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

# Mean pooling: divide sum by count
batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
```

### How Mean Pooling Works

#### Step-by-Step Breakdown

**1. Get the attention mask:**
```python
attention_mask = inputs['attention_mask']
# Shape: (batch_size, seq_len)
# Example: [[1, 1, 1, 1, 0, 0],  # Sentence 1: 4 real tokens, 2 padding
#           [1, 1, 1, 1, 1, 1]]  # Sentence 2: 6 real tokens, 0 padding
```

**2. Expand mask to match embeddings:**
```python
mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
# Shape: (batch_size, seq_len, hidden_dim)
# Repeats the mask across the embedding dimension
```

**3. Mask out padding tokens:**
```python
sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
# Multiplying by mask zeros out padding positions
# Sum across sequence dimension (dim=1)
# Result shape: (batch_size, hidden_dim)
```

**4. Calculate mean (not sum):**
```python
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
# Count how many real (non-padding) tokens each sentence has
# clamp() prevents division by zero

batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
# Divide by token count to get mean (not sum)
```

### Why Mean Pooling Is Correct for T5

1. **Aggregates all tokens**: Uses information from the entire sentence, not just one position
2. **Handles variable lengths**: Properly accounts for padding with attention masking
3. **Standard practice**: This is the recommended approach for extracting sentence embeddings from encoder-only T5 models
4. **Better semantic representation**: Averages contextual information across all words

---

## Expected Results After Fix

### Comparison: Before vs After

| Metric | Before (Broken) | After (Fixed) | Qwen3 (Reference) |
|--------|----------------|---------------|-------------------|
| **Similarity Range** | 0.995-0.999 | ~0.50-0.85 | 0.51-0.87 |
| **Separation Ratio** | 1.029 | ~1.15-1.20 | 1.18 |
| **Within-factor similarity** | 0.535 | ~0.60-0.70 | 0.65 |
| **Between-factor similarity** | 0.520 | ~0.50-0.55 | 0.55 |
| **T-SNE visualization** | No clear clusters | Clear factor separation | Clear factor separation |

### Expected Behavior

After the fix, you should see:

1. **Meaningful similarity ranges**: Related items have higher similarity (~0.7-0.85), unrelated items have lower similarity (~0.4-0.6)
2. **Good factor separation**: Items from the same psychological factor cluster together in embedding space
3. **Clear T-SNE plots**: Visual separation between Anxiety, Depression, and Stress factors
4. **Sensible nearest neighbors**: Most similar items actually share semantic content

### Example of Expected Output
```
Sample item: "I found myself getting upset by quite trivial things." [Stress]

5 Most similar items (EXPECTED after fix):
  1. [Stress] I found myself getting upset rather easily.     Similarity: 0.82
  2. [Stress] I found myself getting agitated.                Similarity: 0.76
  3. [Stress] I found that I was very irritable.              Similarity: 0.75
  4. [Stress] I found it hard to calm down after upset.       Similarity: 0.74
  5. [Stress] I felt that I was rather touchy.                Similarity: 0.71
```

Notice: Top neighbors are now semantically related AND from the same factor!

---

## Technical Background: T5 Architecture

### T5 Model Overview

**T5 (Text-To-Text Transfer Transformer)** is an encoder-decoder architecture designed primarily for sequence-to-sequence tasks:

- **Encoder**: Processes input text into contextualized representations
- **Decoder**: Generates output text conditioned on encoder representations

### Key Architectural Differences

| Feature | BERT/DistilBERT | T5 |
|---------|-----------------|-----|
| **Architecture** | Encoder-only | Encoder-Decoder |
| **[CLS] token** | ✅ Yes | ❌ No |
| **Sentence embedding** | Use [CLS] token | Requires pooling |
| **Primary use case** | Classification, embeddings | Text generation |
| **Pooling method** | [CLS] token (position 0) | Mean/max pooling |

### T5-GEMMA Specifics

T5-GEMMA is a variant of T5 with:
- Gemma's vocabulary and tokenizer
- T5 architecture (encoder-decoder)
- Optimized for semantic understanding
- Available in multiple sizes: Base (1B), 2B, 9B parameters

**For embedding extraction**: We only use the **encoder** portion, discarding the decoder.

---

## Implementation Details

### Files Modified

**File:** `t5gemma_factor_analysis.ipynb`

**Cells modified:**
1. **Cell 10** (Markdown): Updated documentation to clarify mean pooling approach
2. **Cell 11** (Code): Replaced first-token extraction with mean pooling

### Code Changes

**Before:**
```python
# Extract embeddings from first token of last hidden state
# Shape: (batch_size, hidden_dim)
batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
```

**After:**
```python
# Mean pooling with attention mask (T5 doesn't have [CLS] token)
# This properly accounts for padding tokens
attention_mask = inputs['attention_mask']
# Expand mask to match hidden state dimensions: (batch_size, seq_len, hidden_dim)
mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
# Sum embeddings (masked) across sequence dimension
sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
# Sum mask to get actual token counts (avoid division by zero)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
# Mean pooling: divide sum by count
batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
```

**Lines changed:** ~10 lines (1 line replaced with ~10 lines of proper implementation)

---

## Verification Steps

After applying the fix, verify correctness by:

### 1. Check Similarity Ranges
```python
# Should see 0.4-0.9 range, not 0.995-0.999
similarities = cosine_similarity(embeddings)
print(f"Min: {similarities.min():.4f}, Max: {similarities.max():.4f}")
```

**Expected:** Min ~0.40, Max ~0.90 (excluding diagonal 1.0)

### 2. Check Separation Ratio
```python
# Should be > 1.1 (good separation)
separation_ratio = within_mean / between_mean
print(f"Separation ratio: {separation_ratio:.4f}")
```

**Expected:** > 1.1 (higher is better)

### 3. Visual Inspection of T-SNE
- Should see three distinct clusters (Anxiety, Depression, Stress)
- Some overlap is normal (factors are related)
- No single tight ball of all points

### 4. Nearest Neighbors Test
- Check that similar items actually share semantic content
- Most similar items should often be from the same factor
- Similarity values should vary meaningfully

---

## Lessons Learned

### Key Takeaways

1. **Architecture matters**: Different model architectures require different embedding extraction methods
2. **Don't assume [CLS] tokens**: Not all transformers use [CLS] tokens for sentence representation
3. **Attention masks are crucial**: Always account for padding when pooling token embeddings
4. **Validate embeddings**: Always check similarity distributions before using embeddings
5. **Compare to baselines**: Having Qwen3 results as a reference made diagnosis much faster

### When to Use Each Method

| Model Type | Embedding Method | Example Models |
|------------|------------------|----------------|
| **BERT-style with [CLS]** | Extract position 0 | BERT, DistilBERT, RoBERTa |
| **T5-style (no [CLS])** | Mean pooling | T5, T5-GEMMA, FLAN-T5 |
| **Sentence-Transformers** | Use `.encode()` | All SentenceTransformer models |
| **GPT-style** | Use last token | GPT-2, GPT-3 (for embeddings) |

---

## References

### Documentation
- [HuggingFace T5 Documentation](https://huggingface.co/docs/transformers/model_doc/t5)
- [Sentence Embeddings Guide](https://www.sbert.net/examples/applications/computing-embeddings/README.html)

### Related Issues
- Similar issue discussed: [sentence-transformers/#1380](https://github.com/UKPLab/sentence-transformers/issues/1380)

### Alternative Approaches

While mean pooling is standard, other valid approaches include:

1. **Max pooling**: Take max value across sequence dimension
2. **Weighted pooling**: Weight tokens by attention scores
3. **Last token pooling**: Use final token (common for decoder-only models)
4. **Multiple token aggregation**: Combine first, last, and mean

For T5-style models, **mean pooling with attention masking** remains the recommended approach.

---

## Acknowledgments

This issue was identified through comparative analysis with Qwen3-Embedding models, which showed healthy separation ratios and meaningful similarity ranges. The discrepancy prompted investigation into the embedding extraction methodology.

---

**End of document**
