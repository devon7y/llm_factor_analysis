#!/usr/bin/env python3
"""Rerun all AQ subscale-dependent analyses after correcting 3 miscoded items.

Items corrected:
  AQ10: Social_Skills → Attention_Switching
  AQ20: Communication → Imagination
  AQ41: Attention_to_Detail → Imagination

Affected outputs:
  1. AQ-50 within/between separation (semantic + empirical)
  2. AQ-50 Tucker congruence against corrected theoretical subscales
  3. Cross-scale subscale decomposition (R² per TAS×AQ pairing)
  4. SEM Models 1–5 (subscale parcels)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from factor_analyzer import FactorAnalyzer
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/Users/devon7y/VS_Code/LLM_Factor_Analysis")

# Load data
tas_items = pd.read_csv(BASE / "scale_items/TAS-20_items.csv")
aq_items = pd.read_csv(BASE / "scale_items/AQ-50_items.csv")
tas_resp = pd.read_csv(BASE / "scale_responses/TAS-20_data.csv", sep='\t')
aq_resp = pd.read_csv(BASE / "scale_responses/AQ-50_data.csv", sep='\t')
tas_emb = np.load(BASE / "embeddings/TAS-20_items_8B.npz", allow_pickle=True)['embeddings']
aq_emb = np.load(BASE / "embeddings/AQ-50_items_8B.npz", allow_pickle=True)['embeddings']

# Verify corrected subscale counts
print("AQ-50 subscale counts (should be 10 each):")
print(aq_items['factor'].value_counts().sort_index())
assert all(aq_items['factor'].value_counts() == 10), "Not all subscales have 10 items!"

n_tas = len(tas_items)
n_aq = len(aq_items)
tas_factors = tas_items['factor'].values
aq_factors = aq_items['factor'].values
aq_codes = aq_items['code'].values

# Compute similarity matrices
S_aq = cosine_similarity(aq_emb)

# Load empirical correlation matrix (polychoric, precomputed)
R_full = np.load(BASE / "results/cross_scale/R_polychoric_70x70.npy")
R_aq = R_full[n_tas:, n_tas:]

combined_emb = np.vstack([tas_emb, aq_emb])
S_cross = cosine_similarity(tas_emb, aq_emb)

print("\n" + "=" * 70)
print("1. AQ-50 WITHIN/BETWEEN SEPARATION (CORRECTED SUBSCALES)")
print("=" * 70)

for label, mat in [("Embeddings", S_aq), ("Empirical", R_aq)]:
    within, between = [], []
    for i in range(n_aq):
        for j in range(i + 1, n_aq):
            if aq_factors[i] == aq_factors[j]:
                within.append(mat[i, j])
            else:
                between.append(mat[i, j])
    within = np.array(within)
    between = np.array(between)
    m_w, m_b = np.mean(within), np.mean(between)
    sd_w, sd_b = np.std(within, ddof=1), np.std(between, ddof=1)
    pooled_sd = np.sqrt((sd_w**2 + sd_b**2) / 2)
    d = (m_w - m_b) / pooled_sd if pooled_sd > 0 else 0
    t, p = ttest_ind(within, between, equal_var=False)
    print(f"\n  {label}:")
    print(f"    Within:  M = {m_w:.3f}, SD = {sd_w:.3f}, n = {len(within)}")
    print(f"    Between: M = {m_b:.3f}, SD = {sd_b:.3f}, n = {len(between)}")
    print(f"    Cohen's d = {d:.3f}")
    print(f"    t = {t:.3f}, p = {p:.2e}")


print("\n" + "=" * 70)
print("2. AQ-50 TUCKER CONGRUENCE (CORRECTED SUBSCALES)")
print("=" * 70)

def compute_tucker_congruence(loadings, reference):
    num = np.sum(loadings * reference)
    den = np.sqrt(np.sum(loadings**2)) * np.sqrt(np.sum(reference**2))
    return num / den if den != 0 else 0.0

# Theoretical indicator matrix
unique_factors = sorted(set(aq_factors))
theoretical = pd.DataFrame(
    {f: (aq_factors == f).astype(float) for f in unique_factors},
    index=aq_codes
)

datasets = [
    ("Semantic (SFA)", S_aq, None),
    ("Empirical (EFA)", R_aq, aq_resp.values),
]
for label, corr_mat, raw_data in datasets:
    from factor_analyzer.factor_analyzer import calculate_kmo
    kmo_all, kmo = calculate_kmo(corr_mat)

    # Parallel analysis and EFA using is_corr_matrix=True (matches qwen3_efa.py pipeline)
    if raw_data is not None:
        n_obs = raw_data.shape[0]
        fit_matrix = np.corrcoef(raw_data.T)
        obs_eigenvalues = np.sort(np.linalg.eigvalsh(fit_matrix))[::-1]
        random_eigenvalues = []
        for r_idx in range(100):
            rng2 = np.random.default_rng(42 + r_idx)
            random_data = rng2.standard_normal((n_obs, n_aq))
            random_corr = np.corrcoef(random_data.T)
            eigs = np.sort(np.linalg.eigvalsh(random_corr))[::-1]
            random_eigenvalues.append(eigs)
        pa_threshold = np.percentile(np.array(random_eigenvalues), 95, axis=0)
    else:
        fit_matrix = corr_mat
        embedding_dim = aq_emb.shape[1]
        obs_eigenvalues = np.sort(np.linalg.eigvalsh(fit_matrix))[::-1]
        random_eigenvalues = []
        rng = np.random.default_rng(42)
        for _ in range(100):
            random_vectors = rng.standard_normal((n_aq, embedding_dim))
            random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)
            random_sim = random_vectors @ random_vectors.T
            eigs = np.sort(np.linalg.eigvalsh(random_sim))[::-1]
            random_eigenvalues.append(eigs)
        pa_threshold = np.percentile(np.array(random_eigenvalues), 95, axis=0)

    n_factors = max(int(np.sum(obs_eigenvalues > pa_threshold)), 1)

    fa2 = FactorAnalyzer(rotation='oblimin', method='minres', n_factors=n_factors,
                         is_corr_matrix=True, rotation_kwargs={'normalize': True})
    fa2.fit(fit_matrix)
    loadings = pd.DataFrame(fa2.loadings_, index=aq_codes,
                           columns=[f"F{i+1}" for i in range(n_factors)])

    var_explained = fa2.get_factor_variance()
    total_var = np.sum(var_explained[1]) * 100

    print(f"\n  {label}:")
    print(f"    KMO = {kmo:.3f}")
    print(f"    Factors retained: {n_factors}")
    print(f"    Variance explained: {total_var:.1f}%")

    # Tucker congruence: best match per theoretical subscale
    tucker_results = []
    for tf in unique_factors:
        best_phi = -1
        best_ef = None
        for ef in loadings.columns:
            phi = compute_tucker_congruence(loadings[ef].values, theoretical[tf].values)
            if phi > best_phi:
                best_phi = phi
                best_ef = ef
        tucker_results.append((tf, best_ef, best_phi))
        print(f"    {tf}: best match {best_ef}, φ = {best_phi:.3f}")


print("\n" + "=" * 70)
print("3. CROSS-SCALE SUBSCALE DECOMPOSITION (CORRECTED)")
print("=" * 70)

tas_factor_list = sorted(tas_items['factor'].unique())
aq_factor_list = sorted(aq_items['factor'].unique())

subscale_results = []
for tf in tas_factor_list:
    for af in aq_factor_list:
        tas_mask = np.array(tas_factors == tf)
        aq_mask = np.array(aq_factors == af)

        S_block = S_cross[np.ix_(tas_mask, aq_mask)]
        R_block = R_full[np.ix_(np.where(np.concatenate([[True]*0, tas_mask, [False]*n_aq]))[0],
                                np.where(np.concatenate([[False]*n_tas, aq_mask]))[0])]

        # Get the correct R block from the full 70x70 matrix
        tas_indices = np.where(tas_mask)[0]  # indices within TAS (0-19)
        aq_indices = np.where(aq_mask)[0] + n_tas  # indices within full matrix (20-69)
        R_block = R_full[np.ix_(tas_indices, aq_indices)]

        s_flat = S_block.flatten()
        r_flat = R_block.flatten()
        r_corr, p_val = pearsonr(s_flat, r_flat)

        subscale_results.append({
            'TAS': tf, 'AQ': af,
            'mean_S': np.mean(s_flat), 'mean_R': np.mean(r_flat),
            'r': r_corr, 'R2': r_corr**2, 'p': p_val,
            'n_pairs': len(s_flat)
        })

sub_df = pd.DataFrame(subscale_results)
print("\n  Subscale-level S-R alignment:")
print(f"  {'TAS':<5} {'AQ':<22} {'mean_S':>7} {'mean_R':>7} {'R²':>7} {'p':>10} {'n':>5}")
print(f"  {'-'*5} {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*5}")
for _, row in sub_df.sort_values('R2', ascending=False).iterrows():
    print(f"  {row['TAS']:<5} {row['AQ']:<22} {row['mean_S']:>7.3f} {row['mean_R']:>7.3f} {row['R2']:>7.3f} {row['p']:>10.3e} {row['n_pairs']:>5.0f}")


print("\n" + "=" * 70)
print("4. SEM MODELS 1-5 (CORRECTED PARCELS)")
print("=" * 70)

import semopy

# Compute subscale scores
tas_subscales = {}
for factor in sorted(tas_items['factor'].unique()):
    cols = tas_items[tas_items['factor'] == factor]['code'].values
    tas_subscales[factor] = tas_resp[cols].mean(axis=1).values

aq_subscales = {}
for factor in sorted(aq_items['factor'].unique()):
    cols = aq_items[aq_items['factor'] == factor]['code'].values
    aq_subscales[factor] = aq_resp[cols].mean(axis=1).values

subscale_data = {}
for k, v in tas_subscales.items():
    subscale_data[k] = v
for k, v in aq_subscales.items():
    subscale_data[k] = v

subscale_score_df = pd.DataFrame(subscale_data)

rename_map = {
    'Attention_Switching': 'AtnSw',
    'Attention_to_Detail': 'AtnDt',
    'Communication': 'Comm',
    'Imagination': 'Imag',
    'Social_Skills': 'SocSk',
}
sem_df = subscale_score_df.rename(columns=rename_map)
print(f"\n  SEM variables: {sem_df.columns.tolist()}")
print(f"  N = {len(sem_df)}")

def get_latent_corr(model):
    params = model.inspect()
    cov_row = params[(params['op'] == '~~') &
                     (((params['lval'] == 'TAS') & (params['rval'] == 'AQ')) |
                      ((params['lval'] == 'AQ') & (params['rval'] == 'TAS')))]
    if cov_row.empty:
        return None
    cov = float(cov_row['Estimate'].values[0])
    tas_var = params[(params['op'] == '~~') & (params['lval'] == 'TAS') & (params['rval'] == 'TAS')]
    aq_var = params[(params['op'] == '~~') & (params['lval'] == 'AQ') & (params['rval'] == 'AQ')]
    if tas_var.empty or aq_var.empty:
        return None
    return cov / np.sqrt(float(tas_var['Estimate'].values[0]) * float(aq_var['Estimate'].values[0]))

# Model 1: Baseline
model1_spec = """
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
TAS ~~ AQ
"""
m1 = semopy.Model(model1_spec)
m1.fit(sem_df)
r1 = get_latent_corr(m1)
s1 = semopy.calc_stats(m1)
print(f"\n  Model 1 (Baseline): r = {r1:.3f}")
print(f"    CFI = {s1['CFI'].values[0]:.3f}, RMSEA = {s1['RMSEA'].values[0]:.3f}")
chi2_1 = s1['chi2'].values[0]
df_1 = s1['DoF'].values[0]
print(f"    χ² = {chi2_1:.1f}, df = {df_1:.0f}")

# Model 2: DIF free residual covariances
model2_spec = """
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
TAS ~~ AQ
DIF ~~ SocSk + Comm + AtnSw + AtnDt + Imag
"""
m2 = semopy.Model(model2_spec)
m2.fit(sem_df)
r2 = get_latent_corr(m2)
s2 = semopy.calc_stats(m2)
red2 = (1 - r2/r1) * 100
print(f"\n  Model 2 (DIF free residuals): r = {r2:.3f}, reduction = {red2:.1f}%")
print(f"    CFI = {s2['CFI'].values[0]:.3f}, RMSEA = {s2['RMSEA'].values[0]:.3f}")
chi2_2 = s2['chi2'].values[0]
df_2 = s2['DoF'].values[0]
print(f"    χ² = {chi2_2:.1f}, df = {df_2:.0f}")

# Model 3: All free residual covariances
model3_spec = """
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
TAS ~~ AQ
DIF ~~ SocSk + Comm + AtnSw + AtnDt + Imag
DDF ~~ SocSk + Comm + AtnSw + AtnDt + Imag
EOT ~~ SocSk + Comm + AtnSw + AtnDt + Imag
"""
m3 = semopy.Model(model3_spec)
m3.fit(sem_df)
r3 = get_latent_corr(m3)
s3 = semopy.calc_stats(m3)
red3 = (1 - r3/r1) * 100
print(f"\n  Model 3 (All free residuals): r = {r3:.3f}, reduction = {red3:.1f}%")
print(f"    CFI = {s3['CFI'].values[0]:.3f}, RMSEA = {s3['RMSEA'].values[0]:.3f}")
chi2_3 = s3['chi2'].values[0]
df_3 = s3['DoF'].values[0]
print(f"    χ² = {chi2_3:.1f}, df = {df_3:.0f}")

# Model 4: Semantic method factor
# Loadings proportional to each subscale's mean cross-scale semantic similarity
tas_cross_weights = {}
for tf in sorted(tas_items['factor'].unique()):
    tas_mask = tas_factors == tf
    tas_cross_weights[tf] = S_cross[tas_mask, :].mean()

aq_cross_weights = {}
for af in sorted(aq_items['factor'].unique()):
    aq_mask = aq_factors == af
    aq_cross_weights[af] = S_cross[:, aq_mask].mean()

print("\n  Cross-scale semantic weights (for Model 4):")
for k, v in {**tas_cross_weights, **aq_cross_weights}.items():
    print(f"    {k}: {v:.3f}")

rename_back = {v: k for k, v in rename_map.items()}
all_weights = {}
for var_name in ['DIF', 'DDF', 'EOT']:
    all_weights[var_name] = tas_cross_weights[var_name]
for short, long in [('SocSk','Social_Skills'),('Comm','Communication'),
                    ('AtnSw','Attention_Switching'),('AtnDt','Attention_to_Detail'),('Imag','Imagination')]:
    all_weights[short] = aq_cross_weights[long]

mean_w = np.mean(list(all_weights.values()))
sd_w = np.std(list(all_weights.values()))
norm_weights = {k: (v - mean_w) / sd_w for k, v in all_weights.items()}

print("\n  Normalized semantic weights (for Model 4):")
for k, v in norm_weights.items():
    print(f"    {k}: {v:.3f} (raw: {all_weights[k]:.3f})")

loading_lines = [f"{w:.4f}*{var}" for var, w in norm_weights.items()]

model4_spec = f"""
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
Sem =~ {' + '.join(loading_lines)}
TAS ~~ AQ
Sem ~~ TAS
Sem ~~ AQ
"""
try:
    m4 = semopy.Model(model4_spec)
    m4.fit(sem_df)
    r4 = get_latent_corr(m4)
    s4 = semopy.calc_stats(m4)
    red4 = (1 - r4/r1) * 100
    print(f"\n  Model 4 (Semantic method factor): r = {r4:.3f}, reduction = {red4:.1f}%")
    print(f"    CFI = {s4['CFI'].values[0]:.3f}, RMSEA = {s4['RMSEA'].values[0]:.3f}")
    chi2_4 = s4['chi2'].values[0]
    df_4 = s4['DoF'].values[0]
    print(f"    χ² = {chi2_4:.1f}, df = {df_4:.0f}")
except Exception as e:
    print(f"\n  Model 4 failed: {e}")
    r4, red4, chi2_4, df_4 = None, None, None, None

# Model 5c: Embedding-constrained DIF covariances (profile likelihood)
print("\n--- Model 5c: Embedding-constrained DIF covariances ---")

mean_S_pairings = {}
for af in sorted(aq_items['factor'].unique()):
    tas_mask = tas_factors == 'DIF'
    aq_mask = aq_factors == af
    mean_S_pairings[af] = S_cross[np.ix_(tas_mask, aq_mask)].mean()
    print(f"  DIF × {af}: mean S = {mean_S_pairings[af]:.3f}")

aq_short = {'Attention_Switching': 'AtnSw', 'Attention_to_Detail': 'AtnDt',
            'Communication': 'Comm', 'Imagination': 'Imag', 'Social_Skills': 'SocSk'}
weights_5c = {aq_short[af]: w for af, w in mean_S_pairings.items()}

def build_model5c_spec(c, weights):
    lines = [
        "TAS =~ DIF + DDF + EOT",
        "AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag",
        "TAS ~~ AQ",
    ]
    for af_short, w in weights.items():
        val = c * w
        lines.append(f"DIF ~~ {val:.8f}*{af_short}")
    return "\n".join(lines)

c_grid = np.linspace(-0.05, 0.15, 2000)
results_5c = []
for c in c_grid:
    spec = build_model5c_spec(c, weights_5c)
    try:
        m = semopy.Model(spec)
        res = m.fit(sem_df)
        if hasattr(res, 'success') and not res.success:
            continue
        r = get_latent_corr(m)
        st = semopy.calc_stats(m)
        chi2 = st['chi2'].values[0] if 'chi2' in st.columns else None
        if r is not None and chi2 is not None:
            results_5c.append((c, r, chi2))
    except:
        continue

if results_5c:
    results_5c = np.array(results_5c)
    best_idx = np.argmin(results_5c[:, 2])
    best_c = results_5c[best_idx, 0]
    best_r = results_5c[best_idx, 1]
    best_chi2 = results_5c[best_idx, 2]

    def chi2_for_c(c_val):
        spec = build_model5c_spec(c_val, weights_5c)
        try:
            m = semopy.Model(spec)
            m.fit(sem_df)
            st = semopy.calc_stats(m)
            return st['chi2'].values[0]
        except:
            return 1e10

    from scipy.optimize import minimize_scalar
    refined = minimize_scalar(chi2_for_c, bounds=(best_c - 0.005, best_c + 0.005), method='bounded')
    if refined.success:
        best_c = refined.x

    # Refit at optimal c
    spec = build_model5c_spec(best_c, weights_5c)
    m5c = semopy.Model(spec)
    m5c.fit(sem_df)
    r5c = get_latent_corr(m5c)
    s5c = semopy.calc_stats(m5c)
    red5c = (1 - r5c/r1) * 100
    chi2_5c = s5c['chi2'].values[0]
    df_5c = s5c['DoF'].values[0]
    cfi_5c = s5c['CFI'].values[0]
    rmsea_5c = s5c['RMSEA'].values[0]

    print(f"\n  Model 5c (DIF proportional): optimal c = {best_c:.4f}")
    print(f"    r = {r5c:.3f}, reduction = {red5c:.1f}%")
    print(f"    CFI = {cfi_5c:.3f}, RMSEA = {rmsea_5c:.3f}")
    print(f"    χ² = {chi2_5c:.1f}, df = {df_5c:.0f}")
else:
    print("  Model 5c: No valid fits found!")
    r5c, red5c, chi2_5c, df_5c = None, None, None, None

print("\n" + "=" * 70)
print("SUMMARY OF CORRECTED RESULTS")
print("=" * 70)
print(f"\n  Model 1 (Baseline):              r = {r1:.3f}")
print(f"  Model 2 (DIF free):              r = {r2:.3f}  ({red2:.1f}% reduction)")
print(f"  Model 3 (All free):              r = {r3:.3f}  ({red3:.1f}% reduction)")
if r4 is not None:
    print(f"  Model 4 (Semantic factor):        r = {r4:.3f}  ({red4:.1f}% reduction)")
if r5c is not None:
    print(f"  Model 5c (DIF proportional):      r = {r5c:.3f}  ({red5c:.1f}% reduction)")
