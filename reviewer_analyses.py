#!/usr/bin/env python3
"""
Reviewer-requested analyses for the alexithymia–autism SFA paper.

Analysis 1: Dependence-aware inference for cross-scale item-pair regression
  a) Row/column permutation test (MRQAP) for the 20×50 cross-scale block
  b) Two-way clustered standard errors (Cameron-Gelbach-Miller)
  c) Subscale-level permutation p-values

Analysis 2: Crossed random-effects dyadic regression
  R_ij = β₀ + β₁·S_ij + u_i^TAS + u_j^AQ + ε_ij

Analysis 3: SEM with semantic method factor (subscale-level)
  Model 1: Two-factor CFA (TAS, AQ latent), estimate latent correlation
  Model 2: Add cross-scale residual covariances for high-semantic-overlap pairs
  Compare latent TAS-AQ correlation across models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.stats import pearsonr, norm, multivariate_normal
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/Users/devon7y/VS_Code/LLM_Factor_Analysis")

# ─── Load data ───────────────────────────────────────────────────────

tas_items = pd.read_csv(BASE / "scale_items/TAS-20_items.csv")
aq_items = pd.read_csv(BASE / "scale_items/AQ-50_items.csv")
tas_emb = np.load(BASE / "embeddings/TAS-20_items_8B.npz", allow_pickle=True)['embeddings']
aq_emb = np.load(BASE / "embeddings/AQ-50_items_8B.npz", allow_pickle=True)['embeddings']
tas_resp = pd.read_csv(BASE / "scale_responses/TAS-20_data.csv", sep='\t')
aq_resp = pd.read_csv(BASE / "scale_responses/AQ-50_data.csv", sep='\t')

n_tas, n_aq = len(tas_items), len(aq_items)
N = len(tas_resp)

# Semantic similarity matrices
combined_emb = np.vstack([tas_emb, aq_emb])
S_full = cosine_similarity(combined_emb)
S_cross = S_full[:n_tas, n_tas:]  # 20×50

# ─── Polychoric correlation matrix ──────────────────────────────────
# (Reuse from cross_scale_analysis.py)

def _bvn_cdf(x, y, rho):
    if (np.isinf(x) and x < 0) or (np.isinf(y) and y < 0):
        return 0.0
    if np.isinf(x) and x > 0:
        return norm.cdf(y)
    if np.isinf(y) and y > 0:
        return norm.cdf(x)
    return multivariate_normal.cdf([x, y], mean=[0, 0], cov=[[1, rho], [rho, 1]])

def _bvn_rect(xl, xu, yl, yu, rho):
    return (_bvn_cdf(xu, yu, rho) - _bvn_cdf(xl, yu, rho)
            - _bvn_cdf(xu, yl, rho) + _bvn_cdf(xl, yl, rho))

def polychoric_pair(x, y):
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

cache_path = BASE / "results/cross_scale/R_polychoric_70x70.npy"
if cache_path.exists():
    print("Loading cached polychoric matrix...")
    R_full = np.load(cache_path)
    n_total = n_tas + n_aq
    print(f"  ✓ Loaded {R_full.shape} from cache.")
else:
    print("Computing polychoric correlation matrix (70×70)...")
    print("This takes ~5-10 minutes. Please wait.")
    aq_resp_ord = aq_resp.round().astype(int)
    combined_cols = [tas_resp.values[:, i] for i in range(n_tas)] + \
                    [aq_resp_ord.values[:, i] for i in range(n_aq)]
    n_total = n_tas + n_aq
    R_full = np.eye(n_total)
    total_pairs = n_total * (n_total - 1) // 2
    done = 0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            R_full[i, j] = R_full[j, i] = polychoric_pair(combined_cols[i], combined_cols[j])
            done += 1
            if done % 500 == 0:
                print(f"  polychoric: {done}/{total_pairs} pairs...", flush=True)
    print("  ✓ Polychoric matrix complete.")
    np.save(cache_path, R_full)
    print(f"  ✓ Cached to {cache_path}")

R_cross = R_full[:n_tas, n_tas:]  # 20×50

s_flat = S_cross.flatten()
r_flat = R_cross.flatten()
r_observed, _ = pearsonr(s_flat, r_flat)

tas_factors = tas_items['factor'].values
aq_factors = aq_items['factor'].values

print(f"\nBaseline: r(S, R) = {r_observed:.4f}, R² = {r_observed**2:.4f}")
print(f"OLS regression: ", end="")
slope, intercept, _, p_naive, se_naive = stats.linregress(s_flat, r_flat)
print(f"β = {slope:.4f} (SE = {se_naive:.4f}), naive p = {p_naive:.2e}")

# =====================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: DEPENDENCE-AWARE INFERENCE")
print("=" * 70)

# ─── 1a: MRQAP row/column permutation test ──────────────────────────

print("\n--- 1a: MRQAP Row/Column Permutation Test ---")
print("    (Permutes TAS item labels to break row dependence)")

np.random.seed(42)
n_perm = 10000
perm_r = np.zeros(n_perm)

for p in range(n_perm):
    row_perm = np.random.permutation(n_tas)
    S_perm = S_cross[row_perm, :]
    perm_r[p] = pearsonr(S_perm.flatten(), r_flat)[0]

p_row_perm = np.mean(perm_r >= r_observed)
print(f"  Observed r = {r_observed:.4f}")
print(f"  Row-permutation null: mean = {perm_r.mean():.4f}, SD = {perm_r.std():.4f}")
print(f"  p (row permutation, {n_perm:,} iterations) = {p_row_perm:.4f}")

# Also try permuting both rows AND columns
print("\n    (Permuting both TAS and AQ item labels)")
perm_r_both = np.zeros(n_perm)
for p in range(n_perm):
    row_perm = np.random.permutation(n_tas)
    col_perm = np.random.permutation(n_aq)
    S_perm = S_cross[row_perm, :][:, col_perm]
    perm_r_both[p] = pearsonr(S_perm.flatten(), r_flat)[0]

p_both_perm = np.mean(perm_r_both >= r_observed)
print(f"  p (row+col permutation, {n_perm:,} iterations) = {p_both_perm:.4f}")

# ─── 1b: Two-way clustered standard errors ──────────────────────────

print("\n--- 1b: Two-Way Clustered Standard Errors ---")
print("    (Cameron-Gelbach-Miller, clustering by TAS item and AQ item)")

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Build the item-pair dataframe
pair_data = []
for i in range(n_tas):
    for j in range(n_aq):
        pair_data.append({
            'S': S_cross[i, j],
            'R': R_cross[i, j],
            'tas_idx': i,
            'aq_idx': j,
            'tas_factor': tas_factors[i],
            'aq_factor': aq_factors[j],
        })
pair_df = pd.DataFrame(pair_data)

# OLS with two-way clustered SEs
import statsmodels.formula.api as smf

ols_model = smf.ols("R ~ S", data=pair_df).fit()

# One-way clustered by TAS item
clust_tas = ols_model.get_robustcov_results(cov_type='cluster',
                                              groups=pair_df['tas_idx'])
# One-way clustered by AQ item
clust_aq = ols_model.get_robustcov_results(cov_type='cluster',
                                             groups=pair_df['aq_idx'])

# Two-way clustering: V_two = V_tas + V_aq - V_het (HC0)
het = ols_model.get_robustcov_results(cov_type='HC0')

V_tas = clust_tas.cov_params()
V_aq = clust_aq.cov_params()
V_het = het.cov_params()
V_twoway = V_tas + V_aq - V_het

se_twoway_slope = np.sqrt(V_twoway[1, 1])
t_twoway = slope / se_twoway_slope
df_min = min(n_tas, n_aq) - 1  # conservative df
p_twoway = 2 * stats.t.sf(abs(t_twoway), df=df_min)

print(f"  Naive OLS:              β = {slope:.4f}, SE = {se_naive:.4f}, p = {p_naive:.2e}")
print(f"  Clustered (TAS item):   SE = {np.sqrt(V_tas[1,1]):.4f}, p = {clust_tas.pvalues[1]:.4f}")
print(f"  Clustered (AQ item):    SE = {np.sqrt(V_aq[1,1]):.4f}, p = {clust_aq.pvalues[1]:.4f}")
print(f"  Two-way clustered:      SE = {se_twoway_slope:.4f}, t = {t_twoway:.3f}, p = {p_twoway:.4f} (df={df_min})")

# ─── 1c: Subscale-level permutation p-values ────────────────────────

print("\n--- 1c: Subscale-Level Permutation P-Values ---")
print("    (Row permutation within each subscale pairing)")

n_perm_sub = 10000
subscale_results = []

for tf in sorted(tas_items['factor'].unique()):
    for af in sorted(aq_items['factor'].unique()):
        tas_mask = tas_factors == tf
        aq_mask = aq_factors == af

        s_sub = S_cross[np.ix_(tas_mask, aq_mask)].flatten()
        r_sub = R_cross[np.ix_(tas_mask, aq_mask)].flatten()

        if len(s_sub) < 4:
            continue

        r_obs_sub, p_naive_sub = pearsonr(s_sub, r_sub)

        # Row permutation within this subscale block
        n_tas_sub = tas_mask.sum()
        perm_r_sub = np.zeros(n_perm_sub)
        S_sub_block = S_cross[np.ix_(tas_mask, aq_mask)]
        R_sub_block = R_cross[np.ix_(tas_mask, aq_mask)]
        for p in range(n_perm_sub):
            row_p = np.random.permutation(n_tas_sub)
            perm_r_sub[p] = pearsonr(S_sub_block[row_p, :].flatten(),
                                      R_sub_block.flatten())[0]

        p_perm = np.mean(np.abs(perm_r_sub) >= abs(r_obs_sub))

        subscale_results.append({
            'TAS': tf, 'AQ': af,
            'n': len(s_sub),
            'r': r_obs_sub,
            'R2': r_obs_sub**2,
            'p_naive': p_naive_sub,
            'p_perm': p_perm,
        })

sub_df = pd.DataFrame(subscale_results).sort_values('R2', ascending=False)
print(f"\n  {'TAS':>5} × {'AQ':<22} {'n':>4} {'r(S,R)':>7} {'R²':>6} {'p_naive':>10} {'p_perm':>8}")
print("  " + "-" * 75)
for _, row in sub_df.iterrows():
    print(f"  {row['TAS']:>5} × {row['AQ']:<22} {row['n']:>4} "
          f"{row['r']:>7.3f} {row['R2']:>6.3f} {row['p_naive']:>10.4f} {row['p_perm']:>8.4f}")


# =====================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: CROSSED RANDOM-EFFECTS DYADIC REGRESSION")
print("=" * 70)

# R_ij = β₀ + β₁·S_ij + u_i^TAS + u_j^AQ + ε_ij
# Statsmodels MixedLM with variance components for the second grouping

print("\n--- Fitting crossed random-effects model ---")
print("    R_ij = β₀ + β₁·S_ij + u_i^TAS + u_j^AQ + ε_ij")

pair_df['tas_item'] = pair_df['tas_idx'].astype(str)
pair_df['aq_item'] = pair_df['aq_idx'].astype(str)

# Method: use groups=tas_item, add aq_item as a variance component
from statsmodels.regression.mixed_linear_model import MixedLM

vc = {"aq_item": "0 + C(aq_item)"}
mixed_model = MixedLM.from_formula(
    "R ~ S",
    groups="tas_item",
    vc_formula=vc,
    data=pair_df
)
mixed_result = mixed_model.fit(reml=True)

print(mixed_result.summary())

beta_fixed = mixed_result.fe_params['S']
se_fixed = mixed_result.bse_fe['S']
z_val = beta_fixed / se_fixed
p_mixed = 2 * stats.norm.sf(abs(z_val))

print(f"\n  Fixed effect of S:")
print(f"    β = {beta_fixed:.4f}")
print(f"    SE = {se_fixed:.4f}")
print(f"    z = {z_val:.3f}")
print(f"    p = {p_mixed:.4f}")

# Variance components
print(f"\n  Random effects variance:")
if mixed_result.cov_re.size > 0:
    var_tas = mixed_result.cov_re.iloc[0, 0]
else:
    var_tas = 0.0
    print("    (TAS item random effect converged to boundary = 0)")
print(f"    TAS item (group): {var_tas:.6f}")
vc_params = mixed_result.vcomp
var_aq = vc_params[0] if len(vc_params) > 0 else 0.0
print(f"    AQ item (vc):     {var_aq:.6f}")
var_resid = mixed_result.scale
print(f"    Residual:         {var_resid:.6f}")

var_total = var_tas + var_aq + var_resid
print(f"\n  ICC (TAS item): {var_tas/var_total:.4f}")
print(f"  ICC (AQ item):  {var_aq/var_total:.4f}")
print(f"  ICC (residual): {var_resid/var_total:.4f}")

# Also try with groups=aq_item to check if aq clustering is more informative
print("\n--- Alternative: groups=AQ item, vc=TAS item ---")
vc2 = {"tas_item": "0 + C(tas_item)"}
mixed_model2 = MixedLM.from_formula("R ~ S", groups="aq_item", vc_formula=vc2, data=pair_df)
mixed_result2 = mixed_model2.fit(reml=True)
print(mixed_result2.summary())

beta_alt = mixed_result2.fe_params['S']
se_alt = mixed_result2.bse_fe['S']
z_alt = beta_alt / se_alt
p_alt = 2 * stats.norm.sf(abs(z_alt))
print(f"\n  Alternative model: β = {beta_alt:.4f}, SE = {se_alt:.4f}, z = {z_alt:.3f}, p = {p_alt:.4f}")

if mixed_result2.cov_re.size > 0:
    var_aq_alt = mixed_result2.cov_re.iloc[0, 0]
else:
    var_aq_alt = 0.0
var_tas_alt = mixed_result2.vcomp[0] if len(mixed_result2.vcomp) > 0 else 0.0
var_resid_alt = mixed_result2.scale
var_total_alt = var_aq_alt + var_tas_alt + var_resid_alt
print(f"  AQ item var: {var_aq_alt:.6f}, TAS item var: {var_tas_alt:.6f}, residual: {var_resid_alt:.6f}")
if var_total_alt > 0:
    print(f"  ICC: AQ = {var_aq_alt/var_total_alt:.4f}, TAS = {var_tas_alt/var_total_alt:.4f}")

# Use the better-fitting model's SE
se_mixed_best = max(se_fixed, se_alt)
p_mixed_best = max(p_mixed, p_alt)
print(f"\n  Best mixed model SE: {se_mixed_best:.4f}")

# Marginal R²
y_pred_fixed = mixed_result.fe_params['Intercept'] + mixed_result.fe_params['S'] * pair_df['S']
var_fixed = y_pred_fixed.var()
R2_marginal = var_fixed / (var_fixed + var_tas + var_aq + var_resid)
R2_conditional = (var_fixed + var_tas + var_aq) / (var_fixed + var_tas + var_aq + var_resid)
print(f"\n  Marginal R² (fixed effects):     {R2_marginal:.4f}")
print(f"  Conditional R² (fixed + random): {R2_conditional:.4f}")


# =====================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: SEM — LATENT CORRELATION WITH SEMANTIC METHOD FACTOR")
print("=" * 70)

# Compute subscale scores
print("\n--- Computing subscale scores ---")

tas_subscales = {}
for factor in sorted(tas_items['factor'].unique()):
    cols = tas_items[tas_items['factor'] == factor]['code'].values
    tas_subscales[factor] = tas_resp[cols].mean(axis=1).values

aq_subscales = {}
for factor in sorted(aq_items['factor'].unique()):
    cols = aq_items[aq_items['factor'] == factor]['code'].values
    aq_subscales[factor] = aq_resp[cols].mean(axis=1).values

# Build subscale dataframe
subscale_data = {}
for k, v in tas_subscales.items():
    subscale_data[k] = v
for k, v in aq_subscales.items():
    safe_k = k.replace(' ', '_')
    subscale_data[safe_k] = v

subscale_score_df = pd.DataFrame(subscale_data)
print(f"  Subscale scores: {subscale_score_df.shape}")
print(f"  Columns: {subscale_score_df.columns.tolist()}")
print(f"\n  Correlation matrix (subscale means):")
corr_sub = subscale_score_df.corr()
print(corr_sub.round(3).to_string())

# Mean cross-scale semantic similarity per subscale pairing
print("\n--- Mean cross-scale semantic similarity per subscale ---")
mean_S_by_subscale = {}
for tf in sorted(tas_items['factor'].unique()):
    for af in sorted(aq_items['factor'].unique()):
        tas_mask = tas_factors == tf
        aq_mask = aq_factors == af
        mean_S_by_subscale[(tf, af)] = S_cross[np.ix_(tas_mask, aq_mask)].mean()
        print(f"  {tf} × {af}: mean S = {mean_S_by_subscale[(tf, af)]:.3f}")

# Compute each subscale's overall cross-scale semantic weight
# (mean similarity to all subscales on the other scale)
tas_cross_weights = {}
for tf in sorted(tas_items['factor'].unique()):
    tas_mask = tas_factors == tf
    tas_cross_weights[tf] = S_cross[tas_mask, :].mean()

aq_cross_weights = {}
for af in sorted(aq_items['factor'].unique()):
    aq_mask = aq_factors == af
    aq_cross_weights[af] = S_cross[:, aq_mask].mean()

print("\n  Per-subscale cross-scale semantic weight:")
for k, v in {**tas_cross_weights, **aq_cross_weights}.items():
    print(f"    {k}: {v:.3f}")

# ─── SEM with semopy ────────────────────────────────────────────────

import semopy

# Rename columns to be semopy-friendly (no spaces, short)
rename_map = {
    'Attention_Switching': 'AtnSw',
    'Attention_to_Detail': 'AtnDt',
    'Communication': 'Comm',
    'Imagination': 'Imag',
    'Social_Skills': 'SocSk',
}
sem_df = subscale_score_df.rename(columns=rename_map)
print(f"\n  SEM variable names: {sem_df.columns.tolist()}")

# Model 1: Baseline two-factor CFA
print("\n--- Model 1: Baseline Two-Factor CFA ---")

model1_spec = """
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
TAS ~~ AQ
"""

model1 = semopy.Model(model1_spec)
model1_result = model1.fit(sem_df)
print(f"  Fit result: {model1_result}")

stats1 = semopy.calc_stats(model1)
print(f"\n  Model 1 fit statistics:")
for col in stats1.columns:
    print(f"    {col}: {stats1[col].values[0]:.4f}" if isinstance(stats1[col].values[0], float) else f"    {col}: {stats1[col].values[0]}")

est1 = model1.inspect()
print(f"\n  Model 1 parameter estimates:")
print(est1.to_string())

# Extract latent correlation
tas_aq_cov = est1[(est1['lval'] == 'TAS') & (est1['rval'] == 'AQ') & (est1['op'] == '~~')]
if len(tas_aq_cov) == 0:
    tas_aq_cov = est1[(est1['lval'] == 'AQ') & (est1['rval'] == 'TAS') & (est1['op'] == '~~')]

tas_var = est1[(est1['lval'] == 'TAS') & (est1['rval'] == 'TAS') & (est1['op'] == '~~')]
aq_var = est1[(est1['lval'] == 'AQ') & (est1['rval'] == 'AQ') & (est1['op'] == '~~')]

if len(tas_aq_cov) > 0 and len(tas_var) > 0 and len(aq_var) > 0:
    cov_val = tas_aq_cov['Estimate'].values[0]
    tas_sd = np.sqrt(tas_var['Estimate'].values[0])
    aq_sd = np.sqrt(aq_var['Estimate'].values[0])
    latent_corr_1 = cov_val / (tas_sd * aq_sd)
    print(f"\n  *** Model 1 latent TAS-AQ correlation: {latent_corr_1:.4f} ***")
    print(f"      (covariance = {cov_val:.4f}, TAS SD = {tas_sd:.4f}, AQ SD = {aq_sd:.4f})")
else:
    print("\n  WARNING: Could not extract latent correlation from Model 1")
    latent_corr_1 = None

# Model 2: Allow cross-scale residual covariances for DIF × all AQ subscales
print("\n--- Model 2: With Cross-Scale Residual Covariances ---")
print("    (Allowing residual covariances for all 5 DIF × AQ pairings)")

# All 5 DIF × AQ subscale pairings
model2_spec = """
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
TAS ~~ AQ
DIF ~~ SocSk + Comm + AtnSw + AtnDt + Imag
"""

model2 = semopy.Model(model2_spec)
model2_result = model2.fit(sem_df)
print(f"  Fit result: {model2_result}")

stats2 = semopy.calc_stats(model2)
print(f"\n  Model 2 fit statistics:")
for col in stats2.columns:
    print(f"    {col}: {stats2[col].values[0]:.4f}" if isinstance(stats2[col].values[0], float) else f"    {col}: {stats2[col].values[0]}")

est2 = model2.inspect()
print(f"\n  Model 2 parameter estimates:")
print(est2.to_string())

tas_aq_cov2 = est2[(est2['lval'] == 'TAS') & (est2['rval'] == 'AQ') & (est2['op'] == '~~')]
if len(tas_aq_cov2) == 0:
    tas_aq_cov2 = est2[(est2['lval'] == 'AQ') & (est2['rval'] == 'TAS') & (est2['op'] == '~~')]

tas_var2 = est2[(est2['lval'] == 'TAS') & (est2['rval'] == 'TAS') & (est2['op'] == '~~')]
aq_var2 = est2[(est2['lval'] == 'AQ') & (est2['rval'] == 'AQ') & (est2['op'] == '~~')]

if len(tas_aq_cov2) > 0 and len(tas_var2) > 0 and len(aq_var2) > 0:
    cov_val2 = tas_aq_cov2['Estimate'].values[0]
    tas_sd2 = np.sqrt(tas_var2['Estimate'].values[0])
    aq_sd2 = np.sqrt(aq_var2['Estimate'].values[0])
    latent_corr_2 = cov_val2 / (tas_sd2 * aq_sd2)
    print(f"\n  *** Model 2 latent TAS-AQ correlation: {latent_corr_2:.4f} ***")
    print(f"      (covariance = {cov_val2:.4f}, TAS SD = {tas_sd2:.4f}, AQ SD = {aq_sd2:.4f})")
else:
    print("\n  WARNING: Could not extract latent correlation from Model 2")
    latent_corr_2 = None

# Model 3: Allow ALL cross-scale residual covariances (saturated cross-scale)
print("\n--- Model 3: All Cross-Scale Residual Covariances ---")

model3_spec = """
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
TAS ~~ AQ
DIF ~~ SocSk
DIF ~~ Comm
DIF ~~ AtnSw
DIF ~~ AtnDt
DIF ~~ Imag
DDF ~~ SocSk
DDF ~~ Comm
DDF ~~ AtnSw
DDF ~~ AtnDt
DDF ~~ Imag
EOT ~~ SocSk
EOT ~~ Comm
EOT ~~ AtnSw
EOT ~~ AtnDt
EOT ~~ Imag
"""

model3 = semopy.Model(model3_spec)
model3_result = model3.fit(sem_df)
print(f"  Fit result: {model3_result}")

stats3 = semopy.calc_stats(model3)
print(f"\n  Model 3 fit statistics:")
for col in stats3.columns:
    print(f"    {col}: {stats3[col].values[0]:.4f}" if isinstance(stats3[col].values[0], float) else f"    {col}: {stats3[col].values[0]}")

est3 = model3.inspect()
print(f"\n  Model 3 parameter estimates:")
print(est3.to_string())

tas_aq_cov3 = est3[(est3['lval'] == 'TAS') & (est3['rval'] == 'AQ') & (est3['op'] == '~~')]
if len(tas_aq_cov3) == 0:
    tas_aq_cov3 = est3[(est3['lval'] == 'AQ') & (est3['rval'] == 'TAS') & (est3['op'] == '~~')]

tas_var3 = est3[(est3['lval'] == 'TAS') & (est3['rval'] == 'TAS') & (est3['op'] == '~~')]
aq_var3 = est3[(est3['lval'] == 'AQ') & (est3['rval'] == 'AQ') & (est3['op'] == '~~')]

if len(tas_aq_cov3) > 0 and len(tas_var3) > 0 and len(aq_var3) > 0:
    cov_val3 = tas_aq_cov3['Estimate'].values[0]
    tas_sd3 = np.sqrt(abs(tas_var3['Estimate'].values[0]))
    aq_sd3 = np.sqrt(abs(aq_var3['Estimate'].values[0]))
    if tas_sd3 > 0 and aq_sd3 > 0:
        latent_corr_3 = cov_val3 / (tas_sd3 * aq_sd3)
        print(f"\n  *** Model 3 latent TAS-AQ correlation: {latent_corr_3:.4f} ***")
    else:
        latent_corr_3 = None
        print("\n  WARNING: Degenerate variance in Model 3")
else:
    latent_corr_3 = None
    print("\n  WARNING: Could not extract latent correlation from Model 3")


# ─── Model 4: Semantic method factor with fixed loadings ─────────────
print("\n--- Model 4: Semantic Method Factor (Fixed Loadings) ---")
print("    Loadings fixed proportional to each subscale's mean cross-scale S")

# Compute standardized weights (centered, scaled to have SD=1)
all_weights = {}
for k, v in tas_cross_weights.items():
    all_weights[k] = v
aq_rename_rev = {v: k for k, v in rename_map.items()}
for short, long in [('SocSk', 'Social_Skills'), ('Comm', 'Communication'),
                     ('AtnSw', 'Attention_Switching'), ('AtnDt', 'Attention_to_Detail'),
                     ('Imag', 'Imagination')]:
    all_weights[short] = aq_cross_weights[long]

mean_w = np.mean(list(all_weights.values()))
sd_w = np.std(list(all_weights.values()))
norm_weights = {k: (v - mean_w) / sd_w for k, v in all_weights.items()}

print("  Normalized semantic weights:")
for k, v in norm_weights.items():
    print(f"    {k}: {v:.3f} (raw: {all_weights[k]:.3f})")

# In semopy, we can fix loadings using 'start' values and constraints
# Build model with fixed semantic factor loadings
loading_strs = []
for var, w in norm_weights.items():
    loading_strs.append(f"{w:.4f}*{var}")

model4_spec = f"""
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
Sem =~ {' + '.join(loading_strs)}
TAS ~~ AQ
Sem ~~ TAS
Sem ~~ AQ
"""

print(f"\n  Model 4 specification:")
print(model4_spec)

try:
    model4 = semopy.Model(model4_spec)
    model4_result = model4.fit(sem_df)
    print(f"  Fit result: {model4_result}")

    stats4 = semopy.calc_stats(model4)
    print(f"\n  Model 4 fit statistics:")
    for col in stats4.columns:
        val = stats4[col].values[0]
        print(f"    {col}: {val:.4f}" if isinstance(val, float) else f"    {col}: {val}")

    est4 = model4.inspect()
    print(f"\n  Model 4 parameter estimates:")
    print(est4.to_string())

    tas_aq_cov4 = est4[(est4['lval'] == 'TAS') & (est4['rval'] == 'AQ') & (est4['op'] == '~~')]
    if len(tas_aq_cov4) == 0:
        tas_aq_cov4 = est4[(est4['lval'] == 'AQ') & (est4['rval'] == 'TAS') & (est4['op'] == '~~')]

    tas_var4 = est4[(est4['lval'] == 'TAS') & (est4['rval'] == 'TAS') & (est4['op'] == '~~')]
    aq_var4 = est4[(est4['lval'] == 'AQ') & (est4['rval'] == 'AQ') & (est4['op'] == '~~')]

    if len(tas_aq_cov4) > 0 and len(tas_var4) > 0 and len(aq_var4) > 0:
        cov_val4 = tas_aq_cov4['Estimate'].values[0]
        tas_sd4 = np.sqrt(abs(tas_var4['Estimate'].values[0]))
        aq_sd4 = np.sqrt(abs(aq_var4['Estimate'].values[0]))
        if tas_sd4 > 0 and aq_sd4 > 0:
            latent_corr_4 = cov_val4 / (tas_sd4 * aq_sd4)
            print(f"\n  *** Model 4 latent TAS-AQ correlation: {latent_corr_4:.4f} ***")
        else:
            latent_corr_4 = None
    else:
        latent_corr_4 = None
except Exception as e:
    print(f"  Model 4 failed: {e}")
    latent_corr_4 = None


# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY OF ALL ANALYSES")
print("=" * 70)

print("\n--- Analysis 1: Dependence-Aware Inference ---")
print(f"  Cross-scale r(S, R) = {r_observed:.4f}, R² = {r_observed**2:.4f}")
print(f"  OLS β = {slope:.4f}")
print(f"    Naive SE = {se_naive:.4f}, p = {p_naive:.2e}")
print(f"    Two-way clustered SE = {se_twoway_slope:.4f}, p = {p_twoway:.4f}")
print(f"    Row permutation p = {p_row_perm:.4f}")
print(f"    Row+col permutation p = {p_both_perm:.4f}")

print("\n  Subscale-level: top pairings with permutation p-values:")
for _, row in sub_df.head(6).iterrows():
    sig = "*" if row['p_perm'] < .05 else ""
    print(f"    {row['TAS']:>5} × {row['AQ']:<22} R² = {row['R2']:.3f}  "
          f"p_naive = {row['p_naive']:.4f}  p_perm = {row['p_perm']:.4f} {sig}")

print("\n--- Analysis 2: Crossed Random-Effects Model ---")
print(f"  β (semantic similarity) = {beta_fixed:.4f}")
print(f"  SE = {se_fixed:.4f}")
print(f"  p = {p_mixed:.4f}")
print(f"  Marginal R² = {R2_marginal:.4f}")
print(f"  Conditional R² = {R2_conditional:.4f}")
print(f"  Variance partition: TAS = {var_tas/var_total:.3f}, AQ = {var_aq/var_total:.3f}, residual = {var_resid/var_total:.3f}")

print("\n--- Analysis 3: SEM Latent Correlation Comparison ---")
print(f"  Model 1 (baseline):              latent r = {latent_corr_1:.4f}" if latent_corr_1 else "  Model 1: FAILED")
print(f"  Model 2 (all 5 DIF residuals):   latent r = {latent_corr_2:.4f}" if latent_corr_2 else "  Model 2: FAILED")
print(f"  Model 3 (all 15 residuals):       latent r = {latent_corr_3:.4f}" if latent_corr_3 else "  Model 3: FAILED")
print(f"  Model 4 (semantic method factor): latent r = {latent_corr_4:.4f}" if latent_corr_4 else "  Model 4: FAILED")

if latent_corr_1 is not None and latent_corr_2 is not None:
    reduction_2 = (1 - latent_corr_2 / latent_corr_1) * 100
    print(f"\n  Reduction M1 → M2 (DIF residuals): {reduction_2:.1f}%")
if latent_corr_1 is not None and latent_corr_3 is not None:
    reduction_3 = (1 - latent_corr_3 / latent_corr_1) * 100
    print(f"  Reduction M1 → M3 (all residuals): {reduction_3:.1f}%")
if latent_corr_1 is not None and latent_corr_4 is not None:
    reduction_4 = (1 - latent_corr_4 / latent_corr_1) * 100
    print(f"  Reduction M1 → M4 (method factor): {reduction_4:.1f}%")

print("\n  Original partial correlation approach:")
print(f"    Raw r(TAS, AQ) = .314, partial r = .209, reduction = 33.4%")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
