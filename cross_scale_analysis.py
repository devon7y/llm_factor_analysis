#!/usr/bin/env python3
"""
Cross-Scale Semantic Decomposition: TAS-20 × AQ-50
===================================================
Implements the cross-scale analyses from the poster:
  Analysis 2 — Cross-scale semantic overlap
  Analysis 3 — Residualized correlation (what remains after semantics?)

Also produces the combined 70×70 heatmaps for Analysis 1.

Inputs:
  embeddings/TAS-20_items_8B.npz
  embeddings/AQ-50_items_8B.npz
  scale_items/TAS-20_items.csv
  scale_items/AQ-50_items.csv
  scale_responses/TAS-20_data.csv
  scale_responses/AQ-50_data.csv

Outputs (written to results/cross_scale/):
  cross_scale_results.txt          — full console log
  heatmap_S_R_70x70.png            — side-by-side semantic vs empirical heatmaps
  scatter_cross_scale.png          — S_ij vs R_ij for 1000 cross-scale pairs
  bar_raw_vs_residualized.png      — raw vs residualized TAS–AQ correlation
  top_pairs_table.csv              — top semantically similar cross-scale pairs
  subscale_decomposition.csv       — per-subscale-pairing R² values
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.stats import pearsonr, spearmanr, norm, multivariate_normal
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')


# ─── Polychoric / tetrachoric correlation ────────────────────────────

def _bvn_cdf(x, y, rho):
    """Bivariate normal CDF at (x, y) with correlation rho."""
    if (np.isinf(x) and x < 0) or (np.isinf(y) and y < 0):
        return 0.0
    if np.isinf(x) and x > 0:
        return norm.cdf(y)
    if np.isinf(y) and y > 0:
        return norm.cdf(x)
    return multivariate_normal.cdf([x, y], mean=[0, 0], cov=[[1, rho], [rho, 1]])


def _bvn_rect(xl, xu, yl, yu, rho):
    """Probability of bivariate normal in rectangle [xl,xu] × [yl,yu]."""
    return (_bvn_cdf(xu, yu, rho) - _bvn_cdf(xl, yu, rho)
            - _bvn_cdf(xu, yl, rho) + _bvn_cdf(xl, yl, rho))


def polychoric_pair(x, y):
    """Maximum-likelihood polychoric correlation for two ordinal vectors."""
    cats_x = np.sort(np.unique(x))
    cats_y = np.sort(np.unique(y))
    ct = np.zeros((len(cats_x), len(cats_y)))
    x_map = {v: i for i, v in enumerate(cats_x)}
    y_map = {v: i for i, v in enumerate(cats_y)}
    for xi, yi in zip(x, y):
        ct[x_map[xi], y_map[yi]] += 1
    ct /= ct.sum()

    cum_x = np.cumsum(ct.sum(axis=1))
    cum_y = np.cumsum(ct.sum(axis=0))
    thresh_x = norm.ppf(np.clip(cum_x[:-1], 1e-8, 1 - 1e-8))
    thresh_y = norm.ppf(np.clip(cum_y[:-1], 1e-8, 1 - 1e-8))
    tx = np.concatenate([[-np.inf], thresh_x, [np.inf]])
    ty = np.concatenate([[-np.inf], thresh_y, [np.inf]])

    def neg_loglik(rho):
        ll = 0
        for i in range(len(cats_x)):
            for j in range(len(cats_y)):
                if ct[i, j] > 0:
                    p = max(_bvn_rect(tx[i], tx[i+1], ty[j], ty[j+1], rho), 1e-10)
                    ll += ct[i, j] * np.log(p)
        return -ll

    result = minimize_scalar(neg_loglik, bounds=(-0.999, 0.999), method='bounded')
    return result.x


def polychoric_matrix(data_list, labels=None):
    """Compute full polychoric correlation matrix for a list of ordinal arrays."""
    n = len(data_list)
    R = np.eye(n)
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            R[i, j] = R[j, i] = polychoric_pair(data_list[i], data_list[j])
            done += 1
            if done % 200 == 0:
                print(f"    polychoric: {done}/{total} pairs computed...", flush=True)
    return R

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'figure.dpi': 150,
})
sns.set_context("notebook", font_scale=1.1)

RESULTS_DIR = Path("results/cross_scale")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

log_path = RESULTS_DIR / "cross_scale_results.txt"


class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()


sys.stdout = Logger(log_path)

print("=" * 70)
print("CROSS-SCALE SEMANTIC DECOMPOSITION: TAS-20 × AQ-50")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ─── Load scale definitions ───────────────────────────────────────────

tas_items = pd.read_csv("scale_items/TAS-20_items.csv")
aq_items = pd.read_csv("scale_items/AQ-50_items.csv")

print(f"\nTAS-20: {len(tas_items)} items, factors: {tas_items['factor'].unique()}")
print(f"AQ-50:  {len(aq_items)} items, factors: {aq_items['factor'].unique()}")

# ─── Load embeddings ──────────────────────────────────────────────────

tas_emb_data = np.load("embeddings/TAS-20_items_8B.npz", allow_pickle=True)
aq_emb_data = np.load("embeddings/AQ-50_items_8B.npz", allow_pickle=True)

tas_embeddings = tas_emb_data['embeddings']
aq_embeddings = aq_emb_data['embeddings']

print(f"\nTAS embeddings: {tas_embeddings.shape}")
print(f"AQ embeddings:  {aq_embeddings.shape}")

# ─── Load empirical response data ────────────────────────────────────

tas_resp = pd.read_csv("scale_responses/TAS-20_data.csv", sep='\t')
aq_resp = pd.read_csv("scale_responses/AQ-50_data.csv", sep='\t')

print(f"\nEmpirical data: N = {len(tas_resp):,}")
print(f"TAS response range: [{tas_resp.values.min():.1f}, {tas_resp.values.max():.1f}]")
print(f"AQ response range:  [{aq_resp.values.min():.2f}, {aq_resp.values.max():.2f}]")

# ─── Compute combined 70×70 matrices ─────────────────────────────────

print(f"\n{'='*70}")
print("ANALYSIS 1: Combined 70×70 Semantic and Empirical Matrices")
print("=" * 70)

combined_embeddings = np.vstack([tas_embeddings, aq_embeddings])
S_full = cosine_similarity(combined_embeddings)

# Round AQ responses to integers (imputed values are person-means)
aq_resp_ord = aq_resp.round().astype(int)

# Polychoric/tetrachoric correlation matrix (handles mixed ordinal data)
print("\nComputing polychoric correlation matrix (70×70)...")
combined_cols = [tas_resp.values[:, i] for i in range(tas_resp.shape[1])] + \
                [aq_resp_ord.values[:, i] for i in range(aq_resp_ord.shape[1])]
R_full = polychoric_matrix(combined_cols)
print("  ✓ Polychoric matrix complete.")

# Also compute Pearson for comparison
combined_resp = np.hstack([tas_resp.values, aq_resp.values])
R_full_pearson = np.corrcoef(combined_resp.T)

n_tas = len(tas_items)
n_aq = len(aq_items)
n_total = n_tas + n_aq

print(f"Combined semantic matrix S: {S_full.shape}")
print(f"Combined empirical matrix R: {R_full.shape}")

lower_idx = np.tril_indices(n_total, k=-1)
s_lower = S_full[lower_idx]
r_lower = R_full[lower_idx]

r_convergence, p_convergence = pearsonr(s_lower, r_lower)
r_lower_pearson = R_full_pearson[lower_idx]
r_conv_pearson, _ = pearsonr(s_lower, r_lower_pearson)
print(f"\nFull matrix convergence (S vs R, polychoric):")
print(f"  Pearson r = {r_convergence:.4f}, p = {p_convergence:.2e}")
print(f"  N pairs = {len(s_lower):,}")
print(f"  (cf. using Pearson correlations: r = {r_conv_pearson:.4f})")

# Mantel test (permutation-based)
from skbio.stats.distance import mantel
from skbio import DistanceMatrix

S_dist = (1 - S_full).astype(np.float32)
R_dist = (1 - R_full).astype(np.float32)
np.fill_diagonal(S_dist, 0)
np.fill_diagonal(R_dist, 0)

S_dist = (S_dist + S_dist.T) / 2
R_dist = (R_dist + R_dist.T) / 2

S_dm = DistanceMatrix(S_dist)
R_dm = DistanceMatrix(R_dist)
mantel_r, mantel_p, _ = mantel(S_dm, R_dm, method='pearson', permutations=9999)
print(f"  Mantel test: r = {mantel_r:.4f}, p = {mantel_p:.4f}")

# ─── Heatmaps (Figure 1) ─────────────────────────────────────────────

combined_labels = list(tas_items['code']) + list(aq_items['code'])
combined_factors = list(tas_items['factor']) + list(aq_items['factor'])

# Symmetric limits centered at 0 for both matrices
s_abs_max = max(abs(S_full[lower_idx].min()), abs(S_full[lower_idx].max()))
r_abs_max = max(abs(R_full[lower_idx].min()), abs(R_full[lower_idx].max()))

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

im1 = axes[0].imshow(S_full, cmap='RdBu_r', vmin=-s_abs_max, vmax=s_abs_max, aspect='equal')
axes[0].set_title(f"Semantic Similarity (S)\nQwen3-Embedding-8B", fontsize=14)
axes[0].axhline(y=n_tas - 0.5, color='black', linewidth=2)
axes[0].axvline(x=n_tas - 0.5, color='black', linewidth=2)
axes[0].set_xticks([n_tas // 2, n_tas + n_aq // 2])
axes[0].set_xticklabels(['TAS-20', 'AQ-50'], fontsize=12)
axes[0].set_yticks([n_tas // 2, n_tas + n_aq // 2])
axes[0].set_yticklabels(['TAS-20', 'AQ-50'], fontsize=12)
plt.colorbar(im1, ax=axes[0], shrink=0.7, label='Cosine Similarity')

im2 = axes[1].imshow(R_full, cmap='RdBu_r', vmin=-r_abs_max, vmax=r_abs_max, aspect='equal')
axes[1].set_title(f"Empirical Correlation (R, polychoric)\nN = {len(tas_resp):,}", fontsize=14)
axes[1].axhline(y=n_tas - 0.5, color='black', linewidth=2)
axes[1].axvline(x=n_tas - 0.5, color='black', linewidth=2)
axes[1].set_xticks([n_tas // 2, n_tas + n_aq // 2])
axes[1].set_xticklabels(['TAS-20', 'AQ-50'], fontsize=12)
axes[1].set_yticks([n_tas // 2, n_tas + n_aq // 2])
axes[1].set_yticklabels(['TAS-20', 'AQ-50'], fontsize=12)
plt.colorbar(im2, ax=axes[1], shrink=0.7, label='Polychoric r')

fig.suptitle(
    "Combined 70-Item Semantic vs. Empirical Matrices\n"
    f"Matrix convergence: r = {r_convergence:.3f}, Mantel p < .001",
    fontsize=16, y=1.02
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "heatmap_S_R_70x70.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: heatmap_S_R_70x70.png")

# ─── Within-scale convergence ────────────────────────────────────────

S_tas = S_full[:n_tas, :n_tas]
R_tas = R_full[:n_tas, :n_tas]
S_aq = S_full[n_tas:, n_tas:]
R_aq = R_full[n_tas:, n_tas:]

tas_lower = np.tril_indices(n_tas, k=-1)
aq_lower = np.tril_indices(n_aq, k=-1)

r_tas, p_tas = pearsonr(S_tas[tas_lower], R_tas[tas_lower])
r_aq, p_aq = pearsonr(S_aq[aq_lower], R_aq[aq_lower])

print(f"\nWithin-scale convergence:")
print(f"  TAS-20: r = {r_tas:.4f} (p = {p_tas:.2e}, n = {len(S_tas[tas_lower])} pairs)")
print(f"  AQ-50:  r = {r_aq:.4f} (p = {p_aq:.2e}, n = {len(S_aq[aq_lower])} pairs)")

# ─── ANALYSIS 2: Cross-scale decomposition ───────────────────────────

print(f"\n{'='*70}")
print("ANALYSIS 2: Cross-Scale Semantic Overlap")
print("=" * 70)

S_cross = S_full[:n_tas, n_tas:]  # 20 × 50
R_cross = R_full[:n_tas, n_tas:]  # 20 × 50 (polychoric)
R_cross_pearson = R_full_pearson[:n_tas, n_tas:]  # for comparison

s_cross_flat = S_cross.flatten()
r_cross_flat = R_cross.flatten()

r_cross_corr, p_cross_corr = pearsonr(s_cross_flat, r_cross_flat)
rho_cross, p_rho = spearmanr(s_cross_flat, r_cross_flat)

r_cross_corr_pear, _ = pearsonr(s_cross_flat, R_cross_pearson.flatten())

print(f"\nCross-scale item pairs: {len(s_cross_flat)}")
print(f"  Pearson r(S, R_polychoric) = {r_cross_corr:.4f} (p = {p_cross_corr:.2e})")
print(f"  (cf. Pearson r(S, R_pearson) = {r_cross_corr_pear:.4f})")
print(f"  Spearman ρ(S, R) = {rho_cross:.4f} (p = {p_rho:.2e})")
print(f"  R² = {r_cross_corr**2:.4f}")
print(f"  → {r_cross_corr**2*100:.1f}% of cross-scale behavioral covariance")
print(f"    attributable to semantic similarity")

# Regression
slope, intercept, r_val, p_val, se = stats.linregress(s_cross_flat, r_cross_flat)
print(f"\n  Regression: R_ij = {slope:.4f} × S_ij + ({intercept:.4f})")
print(f"  Slope SE = {se:.4f}")

# ─── Subscale-level decomposition ────────────────────────────────────

print(f"\n--- Subscale-Pairing Decomposition ---")
tas_factors = tas_items['factor'].values
aq_factors = aq_items['factor'].values

subscale_results = []
for tf in sorted(tas_items['factor'].unique()):
    for af in sorted(aq_items['factor'].unique()):
        tas_mask = tas_factors == tf
        aq_mask = aq_factors == af

        s_sub = S_cross[np.ix_(tas_mask, aq_mask)].flatten()
        r_sub = R_cross[np.ix_(tas_mask, aq_mask)].flatten()

        if len(s_sub) >= 3:
            r_val_sub, p_val_sub = pearsonr(s_sub, r_sub)
            mean_s = s_sub.mean()
            mean_r = r_sub.mean()
            subscale_results.append({
                'TAS_subscale': tf,
                'AQ_subscale': af,
                'n_pairs': len(s_sub),
                'mean_S': mean_s,
                'mean_R': mean_r,
                'r_SR': r_val_sub,
                'R_squared': r_val_sub ** 2,
                'p_value': p_val_sub,
            })

subscale_df = pd.DataFrame(subscale_results)
subscale_df = subscale_df.sort_values('mean_S', ascending=False)

print(f"\n{'TAS':>5} × {'AQ':<20} {'n':>4} {'mean_S':>7} {'mean_R':>7} {'r(S,R)':>7} {'R²':>6}")
print("-" * 70)
for _, row in subscale_df.iterrows():
    sig = '***' if row['p_value'] < .001 else ('**' if row['p_value'] < .01 else ('*' if row['p_value'] < .05 else ''))
    print(f"{row['TAS_subscale']:>5} × {row['AQ_subscale']:<20} {row['n_pairs']:>4} "
          f"{row['mean_S']:>7.4f} {row['mean_R']:>7.4f} {row['r_SR']:>7.4f} {row['R_squared']:>6.4f} {sig}")

subscale_df.to_csv(RESULTS_DIR / "subscale_decomposition.csv", index=False)

# ─── Top cross-scale pairs ───────────────────────────────────────────

print(f"\n--- Top 15 Most Semantically Similar Cross-Scale Pairs ---\n")

pairs = []
for i in range(n_tas):
    for j in range(n_aq):
        pairs.append({
            'TAS_code': tas_items.iloc[i]['code'],
            'TAS_item': tas_items.iloc[i]['item'],
            'TAS_factor': tas_items.iloc[i]['factor'],
            'AQ_code': aq_items.iloc[j]['code'],
            'AQ_item': aq_items.iloc[j]['item'],
            'AQ_factor': aq_items.iloc[j]['factor'],
            'semantic_sim': S_cross[i, j],
            'empirical_corr': R_cross[i, j],
        })

pairs_df = pd.DataFrame(pairs)
top_pairs = pairs_df.nlargest(15, 'semantic_sim')

for idx, row in top_pairs.iterrows():
    print(f"  S={row['semantic_sim']:.3f}  R={row['empirical_corr']:.3f}  "
          f"{row['TAS_code']}({row['TAS_factor']}) × {row['AQ_code']}({row['AQ_factor']})")
    print(f"    TAS: \"{row['TAS_item'][:80]}\"")
    print(f"    AQ:  \"{row['AQ_item'][:80]}\"")
    print()

top_pairs.to_csv(RESULTS_DIR / "top_pairs_table.csv", index=False)

# ─── Scatter plot (Figure 2) ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))

subscale_pair_labels = []
for i in range(n_tas):
    for j in range(n_aq):
        subscale_pair_labels.append(f"{tas_factors[i]}×{aq_factors[j]}")

subscale_pair_arr = np.array(subscale_pair_labels)

unique_tas_f = sorted(tas_items['factor'].unique())
unique_aq_f = sorted(aq_items['factor'].unique())

highlight_pairings = [
    f"{tf}×{af}"
    for tf in ['DIF', 'DDF']
    for af in ['Communication', 'Social_Skills']
]

colors_highlight = plt.cm.Set2(np.linspace(0, 1, len(highlight_pairings)))
color_map = dict(zip(highlight_pairings, colors_highlight))

other_mask = ~np.isin(subscale_pair_arr, highlight_pairings)
ax.scatter(
    s_cross_flat[other_mask], r_cross_flat[other_mask],
    alpha=0.15, s=15, color='gray', label='Other pairings',
    edgecolors='none'
)

for pairing, color in color_map.items():
    mask = subscale_pair_arr == pairing
    if mask.sum() > 0:
        ax.scatter(
            s_cross_flat[mask], r_cross_flat[mask],
            alpha=0.7, s=40, color=color, label=pairing.replace('_', ' '),
            edgecolors='black', linewidths=0.3
        )

x_line = np.linspace(s_cross_flat.min(), s_cross_flat.max(), 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8)

ax.set_xlabel("Semantic Similarity (cosine)", fontsize=13)
ax.set_ylabel("Empirical Correlation (polychoric r)", fontsize=13)
ax.set_title(
    f"Cross-Scale Decomposition: TAS-20 × AQ-50\n"
    f"r = {r_cross_corr:.3f}, R² = {r_cross_corr**2:.3f}, "
    f"N = 1,000 item pairs",
    fontsize=14
)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "scatter_cross_scale.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: scatter_cross_scale.png")

# ─── ANALYSIS 3: Partial correlation controlling for semantic overlap ─

print(f"\n{'='*70}")
print("ANALYSIS 3: Partial Correlation (Controlling for Semantic Overlap)")
print("=" * 70)

# Per-person semantic overlap score: weight each item by how semantically
# similar it is to items on the other scale (deviation from mean).
# Using centered weights ensures the overlap score captures *differential*
# semantic proximity rather than being a near-perfect proxy for the total score.
# Additionally, use max cross-scale similarity (closest partner) to better
# reflect item-specific overlap rather than diffuse average similarity.

tas_sem_weights_raw = S_cross.max(axis=1)   # (20,) max similarity of each TAS item to any AQ item
aq_sem_weights_raw = S_cross.max(axis=0)    # (50,) max similarity of each AQ item to any TAS item
tas_sem_weights = tas_sem_weights_raw - tas_sem_weights_raw.mean()
aq_sem_weights = aq_sem_weights_raw - aq_sem_weights_raw.mean()

tas_overlap = tas_resp.values @ tas_sem_weights  # (N,) per-person TAS semantic overlap score
aq_overlap = aq_resp.values @ aq_sem_weights     # (N,) per-person AQ semantic overlap score

tas_total = tas_resp.values.mean(axis=1)
aq_total = aq_resp.values.mean(axis=1)

r_raw, p_raw = pearsonr(tas_total, aq_total)
print(f"\nRaw TAS–AQ total score correlation:")
print(f"  r = {r_raw:.4f}, p = {p_raw:.2e}")

# Partial correlation: r(TAS_total, AQ_total | TAS_overlap, AQ_overlap)
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

covariates = add_constant(np.column_stack([tas_overlap, aq_overlap]))

tas_resid = OLS(tas_total, covariates).fit().resid
aq_resid = OLS(aq_total, covariates).fit().resid
r_partial, p_partial = pearsonr(tas_resid, aq_resid)

print(f"\nSemantic overlap scores:")
print(f"  TAS overlap: mean = {tas_overlap.mean():.3f}, SD = {tas_overlap.std():.3f}")
print(f"  AQ overlap:  mean = {aq_overlap.mean():.3f}, SD = {aq_overlap.std():.3f}")
print(f"  r(TAS_total, TAS_overlap) = {pearsonr(tas_total, tas_overlap)[0]:.4f}")
print(f"  r(AQ_total, AQ_overlap)   = {pearsonr(aq_total, aq_overlap)[0]:.4f}")

print(f"\nPartial correlation (controlling for semantic overlap):")
print(f"  r_raw     = {r_raw:.4f}")
print(f"  r_partial = {r_partial:.4f}")
print(f"  Reduction = {(1 - r_partial/r_raw)*100:.1f}%")
print(f"  p_partial = {p_partial:.2e}")

# Bootstrap CIs
print(f"\n--- Bootstrap Confidence Intervals (1000 iterations) ---")
np.random.seed(42)
n_boot = 1000
boot_raw = []
boot_partial = []
boot_r_sq = []

for b in range(n_boot):
    idx = np.random.choice(len(tas_resp), len(tas_resp), replace=True)
    tas_b = tas_resp.values[idx]
    aq_b = aq_resp.values[idx]

    tas_tot_b = tas_b.mean(axis=1)
    aq_tot_b = aq_b.mean(axis=1)
    r_raw_b = pearsonr(tas_tot_b, aq_tot_b)[0]
    boot_raw.append(r_raw_b)

    # Semantic overlap scores (same weights, resampled responses)
    tas_ol_b = tas_b @ tas_sem_weights
    aq_ol_b = aq_b @ aq_sem_weights
    cov_b = add_constant(np.column_stack([tas_ol_b, aq_ol_b]))
    tas_res_b = OLS(tas_tot_b, cov_b).fit().resid
    aq_res_b = OLS(aq_tot_b, cov_b).fit().resid
    r_part_b = pearsonr(tas_res_b, aq_res_b)[0]
    boot_partial.append(r_part_b)

    # Cross-scale R² (using Pearson correlations for speed in bootstrap)
    R_cross_b = np.corrcoef(np.hstack([tas_b, aq_b]).T)[:n_tas, n_tas:]
    r_sr_b = pearsonr(S_cross.flatten(), R_cross_b.flatten())[0]
    boot_r_sq.append(r_sr_b ** 2)

boot_raw = np.array(boot_raw)
boot_partial = np.array(boot_partial)
boot_r_sq = np.array(boot_r_sq)
boot_reduction = 1 - boot_partial / boot_raw

print(f"  Raw r:        {np.mean(boot_raw):.4f} [{np.percentile(boot_raw, 2.5):.4f}, {np.percentile(boot_raw, 97.5):.4f}]")
print(f"  Partial r:    {np.mean(boot_partial):.4f} [{np.percentile(boot_partial, 2.5):.4f}, {np.percentile(boot_partial, 97.5):.4f}]")
print(f"  R² (S→R):     {np.mean(boot_r_sq):.4f} [{np.percentile(boot_r_sq, 2.5):.4f}, {np.percentile(boot_r_sq, 97.5):.4f}]")
print(f"  Reduction %:  {np.mean(boot_reduction)*100:.1f}% [{np.percentile(boot_reduction, 2.5)*100:.1f}%, {np.percentile(boot_reduction, 97.5)*100:.1f}%]")

# ─── Bar chart (Figure 3) ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

bars = ['Raw\nr(TAS, AQ)', 'Partial\n(controlling for\nsemantic overlap)']
vals = [r_raw, r_partial]
ci_low = [
    r_raw - np.percentile(boot_raw, 2.5),
    r_partial - np.percentile(boot_partial, 2.5),
]
ci_high = [
    np.percentile(boot_raw, 97.5) - r_raw,
    np.percentile(boot_partial, 97.5) - r_partial,
]

colors = ['#4C72B0', '#55A868']
bar_positions = [0, 1]

for i, (pos, val, c) in enumerate(zip(bar_positions, vals, colors)):
    ax.bar(pos, val, width=0.5, color=c, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.errorbar(pos, val, yerr=[[ci_low[i]], [ci_high[i]]], fmt='none',
                color='black', capsize=8, linewidth=1.5)
    ax.text(pos, val + ci_high[i] + 0.008, f'r = {val:.3f}', ha='center',
            fontsize=12, fontweight='bold')

reduction_pct = (1 - r_partial / r_raw) * 100
ax.annotate(f'{reduction_pct:.1f}% reduction',
            xy=(0.5, (r_raw + r_partial) / 2), fontsize=11,
            ha='center', style='italic', color='#666666')

ax.set_xticks(bar_positions)
ax.set_xticklabels(bars, fontsize=12)
ax.set_ylabel("Correlation (r)", fontsize=13)
ax.set_title(
    "TAS-20 – AQ-50: Raw vs. Partial Correlation\n"
    f"Controlling for per-person semantic overlap scores",
    fontsize=13
)
ax.set_ylim(0, max(vals) * 1.4)
ax.axhline(y=0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "bar_raw_vs_residualized.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: bar_raw_vs_residualized.png")

# ─── Subscale-level heatmap of mean semantic similarity ──────────────

print(f"\n--- Subscale-Level Mean Semantic Similarity and Empirical Correlation ---\n")

pivot_S = subscale_df.pivot(index='TAS_subscale', columns='AQ_subscale', values='mean_S')
pivot_R = subscale_df.pivot(index='TAS_subscale', columns='AQ_subscale', values='mean_R')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

aq_order = ['Social_Skills', 'Communication', 'Attention_Switching', 'Attention_to_Detail', 'Imagination']
tas_order = ['DIF', 'DDF', 'EOT']

pivot_S = pivot_S.reindex(index=tas_order, columns=aq_order)
pivot_R = pivot_R.reindex(index=tas_order, columns=aq_order)

sns.heatmap(pivot_S, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0],
            vmin=pivot_S.values.min() * 0.9, vmax=pivot_S.values.max() * 1.1,
            linewidths=0.5, cbar_kws={'label': 'Mean Cosine Similarity'})
axes[0].set_title("Mean Semantic Similarity", fontsize=13)
axes[0].set_ylabel("TAS-20 Subscale")
axes[0].set_xlabel("AQ-50 Subscale")
axes[0].set_xticklabels([l.get_text().replace('_', '\n') for l in axes[0].get_xticklabels()],
                         rotation=0, fontsize=9)

sns.heatmap(pivot_R, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1],
            vmin=pivot_R.values.min() * 0.9 if pivot_R.values.min() > 0 else pivot_R.values.min() * 1.1,
            vmax=pivot_R.values.max() * 1.1,
            linewidths=0.5, cbar_kws={'label': 'Mean Polychoric r'})
axes[1].set_title("Mean Empirical Correlation", fontsize=13)
axes[1].set_ylabel("TAS-20 Subscale")
axes[1].set_xlabel("AQ-50 Subscale")
axes[1].set_xticklabels([l.get_text().replace('_', '\n') for l in axes[1].get_xticklabels()],
                         rotation=0, fontsize=9)

fig.suptitle("Subscale-Level Cross-Scale Overlap: TAS-20 × AQ-50", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "subscale_heatmaps.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: subscale_heatmaps.png")

# ─── NEW PLOT: Item-wise S vs R overlay matrix ──────────────────────

fig, ax = plt.subplots(figsize=(14, 6))
S_cross_flat_sorted_idx = np.argsort(S_cross.flatten())[::-1]
ax.bar(range(len(s_cross_flat)), s_cross_flat[S_cross_flat_sorted_idx],
       alpha=0.4, color='steelblue', label='Semantic Similarity (S)')
ax.bar(range(len(r_cross_flat)), r_cross_flat[S_cross_flat_sorted_idx],
       alpha=0.6, color='coral', label='Polychoric Correlation (R)')
ax.set_xlabel("Item Pairs (sorted by semantic similarity)", fontsize=12)
ax.set_ylabel("Value", fontsize=12)
ax.set_title("Cross-Scale Item Pairs: Semantic vs. Empirical\n(sorted by descending S)", fontsize=13)
ax.legend(fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "overlay_S_R_sorted.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: overlay_S_R_sorted.png")

# ─── NEW PLOT: Residual heatmap (R - predicted_R) ──────────────────

R_predicted = slope * S_cross + intercept
R_residual = R_cross - R_predicted

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
res_max = max(abs(R_residual.min()), abs(R_residual.max()))

im1 = axes[0].imshow(S_cross, cmap='YlOrRd', aspect='auto')
axes[0].set_title("Semantic Similarity (S)", fontsize=13)
axes[0].set_ylabel("TAS-20 Items")
axes[0].set_xlabel("AQ-50 Items")
plt.colorbar(im1, ax=axes[0], shrink=0.8)

r_max = max(abs(R_cross.min()), abs(R_cross.max()))
im2 = axes[1].imshow(R_cross, cmap='RdBu_r', vmin=-r_max, vmax=r_max, aspect='auto')
axes[1].set_title("Polychoric Correlation (R)", fontsize=13)
axes[1].set_xlabel("AQ-50 Items")
plt.colorbar(im2, ax=axes[1], shrink=0.8)

im3 = axes[2].imshow(R_residual, cmap='RdBu_r', vmin=-res_max, vmax=res_max, aspect='auto')
axes[2].set_title("Residual (R − predicted R)", fontsize=13)
axes[2].set_xlabel("AQ-50 Items")
plt.colorbar(im3, ax=axes[2], shrink=0.8)

fig.suptitle("Cross-Scale Decomposition: What Semantics Can and Cannot Explain", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "residual_heatmap.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: residual_heatmap.png")

# ─── NEW PLOT: Per-item semantic overlap weight profile ─────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

tas_codes = list(tas_items['code'])
aq_codes = list(aq_items['code'])

# TAS items: max cross-scale similarity
sort_idx_t = np.argsort(tas_sem_weights_raw)[::-1]
axes[0].barh(range(n_tas), tas_sem_weights_raw[sort_idx_t], color='steelblue', alpha=0.8)
axes[0].set_yticks(range(n_tas))
axes[0].set_yticklabels([tas_codes[i] for i in sort_idx_t], fontsize=8)
axes[0].set_xlabel("Max Cosine Similarity to Any AQ-50 Item")
axes[0].set_title("TAS-20: Semantic Proximity to AQ-50", fontsize=13)
axes[0].invert_yaxis()

# AQ items: top 20 by max similarity
sort_idx_a = np.argsort(aq_sem_weights_raw)[::-1][:20]
axes[1].barh(range(20), aq_sem_weights_raw[sort_idx_a], color='coral', alpha=0.8)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels([aq_codes[i] for i in sort_idx_a], fontsize=8)
axes[1].set_xlabel("Max Cosine Similarity to Any TAS-20 Item")
axes[1].set_title("AQ-50: Top 20 Items by Semantic Proximity to TAS-20", fontsize=13)
axes[1].invert_yaxis()

fig.suptitle("Which Items Bridge the Scales? Per-Item Semantic Overlap Weights", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "item_overlap_weights.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: item_overlap_weights.png")

# ─── NEW PLOT: Hexbin density for cross-scale S vs R ────────────────

fig, ax = plt.subplots(figsize=(9, 7))
hb = ax.hexbin(s_cross_flat, r_cross_flat, gridsize=25, cmap='YlOrRd', mincnt=1)
ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.7, label=f'OLS: R = {slope:.3f}S + {intercept:.3f}')
ax.set_xlabel("Semantic Similarity (cosine)", fontsize=13)
ax.set_ylabel("Polychoric Correlation", fontsize=13)
ax.set_title(f"Density of Cross-Scale Item Pairs\nr = {r_cross_corr:.3f}, R² = {r_cross_corr**2:.3f}", fontsize=14)
ax.legend(fontsize=11)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
plt.colorbar(hb, ax=ax, label='Count')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "hexbin_cross_scale.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: hexbin_cross_scale.png")

# ─── NEW PLOT: Factor-colored S vs R with marginal distributions ────

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:, :3])
ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

tas_factor_colors = {'DIF': '#E74C3C', 'DDF': '#3498DB', 'EOT': '#2ECC71'}
for i in range(n_tas):
    tf = tas_factors[i]
    c = tas_factor_colors[tf]
    for j in range(n_aq):
        ax_main.scatter(S_cross[i, j], R_cross[i, j], c=c, alpha=0.35, s=20, edgecolors='none')

for tf, c in tas_factor_colors.items():
    ax_main.scatter([], [], c=c, s=40, label=f'TAS: {tf}')
ax_main.legend(loc='upper left', fontsize=10)
ax_main.set_xlabel("Semantic Similarity", fontsize=12)
ax_main.set_ylabel("Polychoric Correlation", fontsize=12)
ax_main.axhline(y=0, color='gray', linestyle=':', alpha=0.4)

ax_top.hist(s_cross_flat, bins=40, color='steelblue', alpha=0.6, edgecolor='white')
ax_top.set_ylabel("Count")
plt.setp(ax_top.get_xticklabels(), visible=False)

ax_right.hist(r_cross_flat, bins=40, orientation='horizontal', color='coral', alpha=0.6, edgecolor='white')
ax_right.set_xlabel("Count")
plt.setp(ax_right.get_yticklabels(), visible=False)

fig.suptitle("Cross-Scale S vs R with Marginal Distributions\n(colored by TAS-20 factor)", fontsize=14, y=0.95)
plt.savefig(RESULTS_DIR / "marginal_scatter.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: marginal_scatter.png")

# ─── Summary ─────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"\n1. Full 70×70 matrix convergence: r = {r_convergence:.4f} (Mantel p = {mantel_p:.4f})")
print(f"2. Within-scale convergence:")
print(f"   TAS-20: r = {r_tas:.4f}")
print(f"   AQ-50:  r = {r_aq:.4f}")
print(f"3. Cross-scale semantic-empirical correlation: r = {r_cross_corr:.4f}")
print(f"   R² = {r_cross_corr**2:.4f} ({r_cross_corr**2*100:.1f}% of variance)")
print(f"4. Raw TAS–AQ scale correlation: r = {r_raw:.4f}")
print(f"   Partial (controlling for semantic overlap): r = {r_partial:.4f}")
print(f"   Reduction: {reduction_pct:.1f}%")

if r_cross_corr**2 > 0.30:
    print(f"\n→ INTERPRETATION: Large semantic overlap (R² = {r_cross_corr**2:.2f} > .30)")
    print(f"  The TAS–AQ correlation is partly a measurement artifact.")
elif r_cross_corr**2 < 0.15:
    print(f"\n→ INTERPRETATION: Small semantic overlap (R² = {r_cross_corr**2:.2f} < .15)")
    print(f"  The correlation reflects a genuine shared process beyond wording.")
else:
    print(f"\n→ INTERPRETATION: Moderate semantic overlap (R² = {r_cross_corr**2:.2f})")
    print(f"  Semantic similarity explains a meaningful but incomplete portion.")

print(f"\n{'='*70}")
print(f"Analysis complete. Results saved to: {RESULTS_DIR}/")
print(f"{'='*70}")

sys.stdout.flush()
if hasattr(sys.stdout, 'log'):
    sys.stdout.log.close()
