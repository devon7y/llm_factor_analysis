#!/usr/bin/env python3
"""Regenerate AQ-50 figures with corrected subscale assignments.

Affected figures (3 items moved to different subscales):
  - AQ-50_comparison_within_between.png (violin plots with d values)
  - AQ-50_comparison_tucker.png (Tucker congruence heatmaps)
  - AQ-50_comparison_matrices.png (similarity/correlation matrices sorted by subscale)
  - AQ-50_comparison_tsne.png (t-SNE colored by subscale)
  - subscale_heatmaps.png (cross-scale subscale means)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from factor_analyzer import FactorAnalyzer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/Users/devon7y/VS_Code/LLM_Factor_Analysis")
FIGDIR = BASE / "latex-aa" / "figures"

tas_items = pd.read_csv(BASE / "scale_items/TAS-20_items.csv")
aq_items = pd.read_csv(BASE / "scale_items/AQ-50_items.csv")
tas_emb = np.load(BASE / "embeddings/TAS-20_items_8B.npz", allow_pickle=True)['embeddings']
aq_emb = np.load(BASE / "embeddings/AQ-50_items_8B.npz", allow_pickle=True)['embeddings']
tas_resp = pd.read_csv(BASE / "scale_responses/TAS-20_data.csv", sep='\t')
aq_resp = pd.read_csv(BASE / "scale_responses/AQ-50_data.csv", sep='\t')

n_tas = len(tas_items)
n_aq = len(aq_items)
aq_factors = aq_items['factor'].values
aq_codes = aq_items['code'].values
tas_factors = tas_items['factor'].values

S_aq = cosine_similarity(aq_emb)
R_full = np.load(BASE / "results/cross_scale/R_polychoric_70x70.npy")
R_aq = R_full[n_tas:, n_tas:]
S_cross = cosine_similarity(tas_emb, aq_emb)

assert all(aq_items['factor'].value_counts() == 10), "Subscale counts not 10 each!"
print(f"AQ-50 subscale counts:\n{aq_items['factor'].value_counts().sort_index()}")

SUBSCALE_COLORS = {
    'Social_Skills': '#e74c3c',
    'Communication': '#3498db',
    'Attention_Switching': '#2ecc71',
    'Attention_to_Detail': '#9b59b6',
    'Imagination': '#f39c12',
}
AQ_ORDER = ['Social_Skills', 'Communication', 'Attention_Switching',
            'Attention_to_Detail', 'Imagination']

# ─── 1. Within/Between Violin Plot ─────────────────────────────────

print("\n1. Generating within/between violin plot...")

emb_within, emb_between = [], []
emp_within, emp_between = [], []
for i in range(n_aq):
    for j in range(i + 1, n_aq):
        if aq_factors[i] == aq_factors[j]:
            emb_within.append(S_aq[i, j])
            emp_within.append(R_aq[i, j])
        else:
            emb_between.append(S_aq[i, j])
            emp_between.append(R_aq[i, j])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

emp_t, emp_p = ttest_ind(emp_within, emp_between, equal_var=False)
emp_mw, emp_mb = np.mean(emp_within), np.mean(emp_between)
emp_d = (emp_mw - emp_mb) / np.sqrt((np.std(emp_within)**2 + np.std(emp_between)**2) / 2)

emb_t, emb_p = ttest_ind(emb_within, emb_between, equal_var=False)
emb_mw, emb_mb = np.mean(emb_within), np.mean(emb_between)
emb_d = (emb_mw - emb_mb) / np.sqrt((np.std(emb_within)**2 + np.std(emb_between)**2) / 2)

emp_data = pd.DataFrame({
    'value': emp_within + emp_between,
    'type': ['Within'] * len(emp_within) + ['Between'] * len(emp_between)
})
sns.violinplot(data=emp_data, x='type', y='value', hue='type', ax=ax1,
               palette=['#2ecc71', '#e74c3c'], legend=False)
ax1.plot([0], [emp_mw], 'D', color='darkgreen', markersize=10)
ax1.plot([1], [emp_mb], 'D', color='darkred', markersize=10)
y_offset = (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.07
ax1.text(0, ax1.get_ylim()[0] - y_offset, f'n={len(emp_within)}', ha='center', va='top')
ax1.text(1, ax1.get_ylim()[0] - y_offset, f'n={len(emp_between)}', ha='center', va='top')
title1 = f'Within vs Between (Human Responses)\nd = {emp_d:.3f}, p < .001' if emp_p < 0.001 else f'Within vs Between (Human Responses)\nd = {emp_d:.3f}, p = {emp_p:.3f}'
ax1.set_title(title1, fontweight='bold')
ax1.set_ylabel('Correlation')
ax1.set_xlabel('')
ax1.grid(True, alpha=0.3, axis='y')

emb_data = pd.DataFrame({
    'value': emb_within + emb_between,
    'type': ['Within'] * len(emb_within) + ['Between'] * len(emb_between)
})
sns.violinplot(data=emb_data, x='type', y='value', hue='type', ax=ax2,
               palette=['#2ecc71', '#e74c3c'], legend=False)
ax2.plot([0], [emb_mw], 'D', color='darkgreen', markersize=10)
ax2.plot([1], [emb_mb], 'D', color='darkred', markersize=10)
y_offset = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.07
ax2.text(0, ax2.get_ylim()[0] - y_offset, f'n={len(emb_within)}', ha='center', va='top')
ax2.text(1, ax2.get_ylim()[0] - y_offset, f'n={len(emb_between)}', ha='center', va='top')
title2 = f'Within vs Between (Embeddings)\nd = {emb_d:.3f}, p < .001' if emb_p < 0.001 else f'Within vs Between (Embeddings)\nd = {emb_d:.3f}, p = {emb_p:.3f}'
ax2.set_title(title2, fontweight='bold')
ax2.set_ylabel('Cosine Similarity')
ax2.set_xlabel('')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGDIR / "AQ-50_comparison_within_between.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  d(emp) = {emp_d:.3f}, d(emb) = {emb_d:.3f}")
print(f"  Saved: AQ-50_comparison_within_between.png")


# ─── 2. Tucker Congruence Heatmap ──────────────────────────────────

print("\n2. Generating Tucker congruence heatmap...")

def compute_tucker(loadings, reference):
    num = np.sum(loadings * reference)
    den = np.sqrt(np.sum(loadings**2)) * np.sqrt(np.sum(reference**2))
    return num / den if den != 0 else 0.0

unique_factors = sorted(set(aq_factors))
theoretical = pd.DataFrame(
    {f: (aq_factors == f).astype(float) for f in unique_factors},
    index=aq_codes
)

def embedding_parallel_analysis(sim_matrix, embeddings, n_iter=100, percentile=95):
    rng = np.random.default_rng(42)
    n_items, embedding_dim = embeddings.shape
    obs_eigenvalues = np.sort(np.linalg.eigvalsh(sim_matrix))[::-1]
    random_eigenvalues = []
    for _ in range(n_iter):
        random_vectors = rng.standard_normal((n_items, embedding_dim))
        random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)
        random_sim = random_vectors @ random_vectors.T
        eigs = np.sort(np.linalg.eigvalsh(random_sim))[::-1]
        random_eigenvalues.append(eigs)
    thresholds = np.percentile(np.array(random_eigenvalues), percentile, axis=0)
    return max(int(np.sum(obs_eigenvalues > thresholds)), 1), obs_eigenvalues, thresholds

def compute_empirical_parallel_analysis(corr_matrix, n_obs, n_iter=100, percentile=95, random_state=42):
    n_items = corr_matrix.shape[0]
    obs_eigenvalues = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]
    random_eigenvalues = []
    for i in range(n_iter):
        rng = np.random.default_rng(random_state + i)
        random_data = rng.standard_normal((n_obs, n_items))
        random_corr = np.corrcoef(random_data.T)
        eigs = np.sort(np.linalg.eigvalsh(random_corr))[::-1]
        random_eigenvalues.append(eigs)
    thresholds = np.percentile(np.array(random_eigenvalues), percentile, axis=0)
    return max(int(np.sum(obs_eigenvalues > thresholds)), 1), obs_eigenvalues, thresholds

def run_efa_and_tucker(label, corr_mat, raw_data, embeddings=None):
    if raw_data is not None:
        n_obs = raw_data.shape[0]
        emp_corr = np.corrcoef(raw_data.T)
        n_factors, ev, pa_threshold = compute_empirical_parallel_analysis(
            emp_corr, n_obs=n_obs, n_iter=100, random_state=42)
        fit_matrix = emp_corr
    else:
        n_factors, ev, pa_threshold = embedding_parallel_analysis(corr_mat, embeddings)
        fit_matrix = corr_mat

    fa2 = FactorAnalyzer(rotation='oblimin', method='minres', n_factors=n_factors,
                         is_corr_matrix=True, rotation_kwargs={'normalize': True})
    fa2.fit(fit_matrix)
    loadings = pd.DataFrame(fa2.loadings_, index=aq_codes,
                            columns=[f"F{i+1}" for i in range(n_factors)])

    tucker_matrix = pd.DataFrame(index=loadings.columns, columns=unique_factors, dtype=float)
    tucker_best = []
    for tf in unique_factors:
        for ef in loadings.columns:
            tucker_matrix.loc[ef, tf] = compute_tucker(loadings[ef].values, theoretical[tf].values)

    for ef in loadings.columns:
        best_tf = tucker_matrix.loc[ef].idxmax()
        best_phi = tucker_matrix.loc[ef].max()
        tucker_best.append({'extracted_factor': ef, 'best_match': best_tf, 'tucker_phi': best_phi})

    tucker_best = pd.DataFrame(tucker_best)
    print(f"  {label}: {n_factors} factors retained")
    for _, row in tucker_best.iterrows():
        print(f"    {row['extracted_factor']}: best match {row['best_match']}, φ = {row['tucker_phi']:.3f}")

    return tucker_matrix, tucker_best, n_factors

sem_tucker, sem_best, sem_nf = run_efa_and_tucker("Semantic (SFA)", S_aq, None, embeddings=aq_emb)
emp_tucker, emp_best, emp_nf = run_efa_and_tucker("Empirical (EFA)", R_aq, aq_resp.values)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.5)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.5)

sns.heatmap(emp_tucker.values.astype(float), annot=True, annot_kws={'fontsize': 15}, fmt='.3f',
            cmap='YlGnBu', xticklabels=emp_tucker.columns, yticklabels=emp_tucker.index,
            ax=ax1, vmin=0, vmax=1, cbar_ax=cax1, cbar_kws={'label': 'Tucker φ'})
ax1.set_title('Tucker Congruence (Human Responses)', fontweight='bold')
ax1.set_xlabel('Theoretical Factors')
ax1.set_ylabel('Extracted Factors')

sns.heatmap(sem_tucker.values.astype(float), annot=True, annot_kws={'fontsize': 15}, fmt='.3f',
            cmap='YlGnBu', xticklabels=sem_tucker.columns, yticklabels=sem_tucker.index,
            ax=ax2, vmin=0, vmax=1, cbar_ax=cax2, cbar_kws={'label': 'Tucker φ'})
ax2.set_title('Tucker Congruence (Embeddings)', fontweight='bold')
ax2.set_xlabel('Theoretical Factors')
ax2.set_ylabel('Extracted Factors')

plt.tight_layout()
plt.savefig(FIGDIR / "AQ-50_comparison_tucker.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: AQ-50_comparison_tucker.png")


# ─── 3. Matrices Sorted by Subscale ────────────────────────────────

print("\n3. Generating sorted matrices...")

sort_idx = np.argsort([AQ_ORDER.index(f) for f in aq_factors])
S_sorted = S_aq[np.ix_(sort_idx, sort_idx)]
R_sorted = R_aq[np.ix_(sort_idx, sort_idx)]
sorted_factors = aq_factors[sort_idx]
sorted_codes = aq_codes[sort_idx]

boundaries = []
for i, f in enumerate(AQ_ORDER):
    count = np.sum(sorted_factors == f)
    boundaries.append(count)
cum_boundaries = np.cumsum(boundaries)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.5)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.5)

sns.heatmap(R_sorted, cmap='RdBu_r', center=0, ax=ax1, cbar_ax=cax1,
            xticklabels=sorted_codes, yticklabels=sorted_codes,
            cbar_kws={'label': 'Polychoric r'})
ax1.set_title('Empirical Correlations', fontweight='bold')
for b in cum_boundaries[:-1]:
    ax1.axhline(y=b, color='black', linewidth=1.5)
    ax1.axvline(x=b, color='black', linewidth=1.5)
ax1.tick_params(axis='both', labelsize=6)

sns.heatmap(S_sorted, cmap='RdBu_r', center=0.5, ax=ax2, cbar_ax=cax2,
            xticklabels=sorted_codes, yticklabels=sorted_codes,
            cbar_kws={'label': 'Cosine Similarity'})
ax2.set_title('Embedding Similarity', fontweight='bold')
for b in cum_boundaries[:-1]:
    ax2.axhline(y=b, color='black', linewidth=1.5)
    ax2.axvline(x=b, color='black', linewidth=1.5)
ax2.tick_params(axis='both', labelsize=6)

plt.tight_layout()
plt.savefig(FIGDIR / "AQ-50_comparison_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: AQ-50_comparison_matrices.png")


# ─── 4. t-SNE Colored by Subscale ──────────────────────────────────

print("\n4. Generating t-SNE plots...")

emp_dist = 1 - R_aq
np.fill_diagonal(emp_dist, 0)
emp_dist = (emp_dist + emp_dist.T) / 2
emp_dist = np.clip(emp_dist, 0, None)

emb_dist = 1 - S_aq
np.fill_diagonal(emb_dist, 0)
emb_dist = (emb_dist + emb_dist.T) / 2
emb_dist = np.clip(emb_dist, 0, None)

tsne_emp = TSNE(n_components=2, perplexity=10, metric='precomputed', random_state=42, init='random')
emp_2d = tsne_emp.fit_transform(emp_dist)

tsne_emb = TSNE(n_components=2, perplexity=10, metric='precomputed', random_state=42, init='random')
emb_2d = tsne_emb.fit_transform(emb_dist)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

for factor in AQ_ORDER:
    mask = aq_factors == factor
    label = factor.replace('_', ' ')
    color = SUBSCALE_COLORS[factor]
    ax1.scatter(emp_2d[mask, 0], emp_2d[mask, 1], c=color, label=label, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax2.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, label=label, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)

for i in range(n_aq):
    ax1.annotate(aq_codes[i], (emp_2d[i, 0], emp_2d[i, 1]), fontsize=5, alpha=0.6, ha='center', va='bottom')
    ax2.annotate(aq_codes[i], (emb_2d[i, 0], emb_2d[i, 1]), fontsize=5, alpha=0.6, ha='center', va='bottom')

ax1.set_title('t-SNE: Empirical Response Space', fontweight='bold')
ax1.legend(fontsize=8, loc='best')
ax1.set_xlabel('t-SNE 1')
ax1.set_ylabel('t-SNE 2')

ax2.set_title('t-SNE: Embedding Space', fontweight='bold')
ax2.legend(fontsize=8, loc='best')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig(FIGDIR / "AQ-50_comparison_tsne.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: AQ-50_comparison_tsne.png")


# ─── 5. Cross-Scale Subscale Heatmaps ──────────────────────────────

print("\n5. Generating cross-scale subscale heatmaps...")

tas_factor_list = sorted(tas_items['factor'].unique())
aq_factor_list = sorted(aq_items['factor'].unique())

subscale_results = []
for tf in tas_factor_list:
    for af in aq_factor_list:
        tas_mask = tas_factors == tf
        aq_mask = aq_factors == af
        S_block = S_cross[np.ix_(tas_mask, aq_mask)]
        tas_indices = np.where(tas_mask)[0]
        aq_indices = np.where(aq_mask)[0] + n_tas
        R_block = R_full[np.ix_(tas_indices, aq_indices)]
        subscale_results.append({
            'TAS_subscale': tf, 'AQ_subscale': af,
            'mean_S': S_block.mean(), 'mean_R': R_block.mean()
        })

sub_df = pd.DataFrame(subscale_results)

pivot_S = sub_df.pivot(index='TAS_subscale', columns='AQ_subscale', values='mean_S')
pivot_R = sub_df.pivot(index='TAS_subscale', columns='AQ_subscale', values='mean_R')

tas_order = ['DIF', 'DDF', 'EOT']
aq_order_long = ['Social_Skills', 'Communication', 'Attention_Switching',
                 'Attention_to_Detail', 'Imagination']
pivot_S = pivot_S.reindex(index=tas_order, columns=aq_order_long)
pivot_R = pivot_R.reindex(index=tas_order, columns=aq_order_long)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

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
plt.savefig(FIGDIR / "subscale_heatmaps.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: subscale_heatmaps.png")

print("\nAll figures regenerated successfully.")
