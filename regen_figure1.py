#!/usr/bin/env python3
"""Regenerate Figure 1 (70x70 heatmap) with empirical on left, semantic on right."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("/Users/devon7y/VS_Code/LLM_Factor_Analysis")

tas_items = pd.read_csv(BASE / "scale_items/TAS-20_items.csv")
aq_items = pd.read_csv(BASE / "scale_items/AQ-50_items.csv")
tas_emb = np.load(BASE / "embeddings/TAS-20_items_8B.npz", allow_pickle=True)['embeddings']
aq_emb = np.load(BASE / "embeddings/AQ-50_items_8B.npz", allow_pickle=True)['embeddings']
tas_resp = pd.read_csv(BASE / "scale_responses/TAS-20_data.csv", sep='\t')

n_tas = len(tas_items)
n_aq = len(aq_items)
n_total = n_tas + n_aq
N = len(tas_resp)

combined_emb = np.vstack([tas_emb, aq_emb])
S_full = cosine_similarity(combined_emb)

R_full = np.load(BASE / "results/cross_scale/R_polychoric_70x70.npy")

lower_idx = np.tril_indices(n_total, k=-1)
r_convergence, _ = pearsonr(S_full[lower_idx], R_full[lower_idx])

s_abs_max = max(abs(S_full[lower_idx].min()), abs(S_full[lower_idx].max()))
r_abs_max = max(abs(R_full[lower_idx].min()), abs(R_full[lower_idx].max()))

# Add subscale boundary lines
tas_factor_counts = tas_items['factor'].value_counts(sort=False)
aq_factor_counts = aq_items['factor'].value_counts(sort=False)

# Compute subscale boundaries for divider lines
tas_boundaries = []
pos = 0
for f in tas_items['factor'].unique():
    n = (tas_items['factor'] == f).sum()
    pos += n
    if pos < n_tas:
        tas_boundaries.append(pos - 0.5)

aq_boundaries = []
pos = 0
for f in aq_items['factor'].unique():
    n = (aq_items['factor'] == f).sum()
    pos += n
    if pos < n_aq:
        aq_boundaries.append(n_tas + pos - 0.5)

all_boundaries = tas_boundaries + [n_tas - 0.5] + aq_boundaries

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# LEFT: Empirical (R)
im1 = axes[0].imshow(R_full, cmap='RdBu_r', vmin=-r_abs_max, vmax=r_abs_max, aspect='equal')
axes[0].set_title(f"Empirical Correlation (R, polychoric)\nN = {N:,}", fontsize=14)
axes[0].axhline(y=n_tas - 0.5, color='black', linewidth=2)
axes[0].axvline(x=n_tas - 0.5, color='black', linewidth=2)
for b in tas_boundaries + aq_boundaries:
    axes[0].axhline(y=b, color='black', linewidth=0.5, alpha=0.5)
    axes[0].axvline(x=b, color='black', linewidth=0.5, alpha=0.5)
axes[0].set_xticks([n_tas // 2, n_tas + n_aq // 2])
axes[0].set_xticklabels(['TAS-20', 'AQ-50'], fontsize=12)
axes[0].set_yticks([n_tas // 2, n_tas + n_aq // 2])
axes[0].set_yticklabels(['TAS-20', 'AQ-50'], fontsize=12)
plt.colorbar(im1, ax=axes[0], shrink=0.7, label='Polychoric r')

# RIGHT: Semantic (S)
im2 = axes[1].imshow(S_full, cmap='RdBu_r', vmin=-s_abs_max, vmax=s_abs_max, aspect='equal')
axes[1].set_title(f"Semantic Similarity (S)\nQwen3-Embedding-8B", fontsize=14)
axes[1].axhline(y=n_tas - 0.5, color='black', linewidth=2)
axes[1].axvline(x=n_tas - 0.5, color='black', linewidth=2)
for b in tas_boundaries + aq_boundaries:
    axes[1].axhline(y=b, color='black', linewidth=0.5, alpha=0.5)
    axes[1].axvline(x=b, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_xticks([n_tas // 2, n_tas + n_aq // 2])
axes[1].set_xticklabels(['TAS-20', 'AQ-50'], fontsize=12)
axes[1].set_yticks([n_tas // 2, n_tas + n_aq // 2])
axes[1].set_yticklabels(['TAS-20', 'AQ-50'], fontsize=12)
plt.colorbar(im2, ax=axes[1], shrink=0.7, label='Cosine Similarity')

fig.suptitle(
    "Combined 70-Item Semantic vs. Empirical Matrices\n"
    f"Matrix convergence: r = {r_convergence:.3f}, Mantel p < .001",
    fontsize=16, y=1.02
)
plt.tight_layout()

out_path = BASE / "results/cross_scale/heatmap_S_R_70x70.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# Also copy to latex figures
import shutil
latex_fig = BASE / "latex-aa/figures/heatmap_S_R_70x70.png"
shutil.copy2(out_path, latex_fig)
print(f"Copied to: {latex_fig}")
