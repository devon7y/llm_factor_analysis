# -*- coding: utf-8 -*-
"""
LLM Factor Analysis - Personality Items
Extracts embeddings from NEO personality items using DistilBERT
and compares predicted similarities with observed correlations.
"""

import os
import sys

print("Starting LLM Factor Analysis...")

# Check for required data files
required_files = ['NEO_items.csv', 'item_corrs.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"Error: Missing required data files: {', '.join(missing_files)}")
    print("Please ensure these files are in the current working directory.")
    sys.exit(1)

print("Data files found. Loading dependencies (this may take a moment)...")

"""## Processing data"""

import pandas as pd
import numpy as np
print("  - pandas and numpy loaded")

from datasets import Dataset
print("  - datasets loaded")

from transformers import AutoTokenizer
print("  - transformers loaded")

print("Loading NEO personality items...")
neo_items = pd.read_csv('NEO_items.csv', usecols=['item', 'construct'])
print(f"Loaded {len(neo_items)} items")

# Converting into a HuggingFace dataset
dat = Dataset.from_pandas(neo_items)

# Loading the tokenizer
print("\nLoading tokenizer...")
model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print(f'Vocabulary size: {tokenizer.vocab_size}, max context length: {tokenizer.model_max_length}')

# Tokenizing the text
print("\nTokenizing items...")
batch_tokenizer = lambda x: tokenizer(x['item'], padding=True, truncation=True)
dat = dat.map(batch_tokenizer, batched=True, batch_size=None)
print(f"Sample tokenization: {[tokenizer.decode(id) for id in dat['input_ids'][0]]}")

# Setting the format of the dataset to torch tensors for passing to the model
dat.set_format('torch', columns=['input_ids', 'attention_mask'])
print(f"Dataset shape: {np.array(dat['input_ids']).shape}")

"""# Feature extraction"""

print("\nLoading PyTorch (this may take 10-30 seconds on first run)...")
import torch
print("  - torch loaded")

from transformers import AutoModel

# Loading the model and moving it to the GPU if available
print("\nDetecting available device...")
if torch.cuda.is_available():  # for nvidia GPUs
    device = torch.device('cuda')
    print("Using CUDA GPU")
elif torch.backends.mps.is_available(): # for Apple Metal Performance Shaders (MPS) GPUs
    device = torch.device('mps')
    print("Using Apple MPS GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Loading the model
print("\nLoading DistilBERT model...")
model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
print(f'Model inputs: {tokenizer.model_input_names}')

def extract_features(batch):
    """Extract features from a batch of items"""
    inputs = {k:v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


print("\nExtracting features from items...")
dat = dat.map(extract_features, batched=True, batch_size=8)
print(f"Feature extraction complete. Shape: {np.array(dat['hidden_state']).shape}")

print("\nDataset contents:")
print(dat)

print("\nFirst few rows:")
print(dat[:5])

print("\nColumn names:")
print(dat.column_names)

print("\nFirst hidden_state shape:")
print(f"Shape: {dat[0]['hidden_state'].shape if hasattr(dat[0]['hidden_state'], 'shape') else len(dat[0]['hidden_state'])}")

print("\nFirst row full details:")
print(dat[0])
