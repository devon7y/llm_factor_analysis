#!/usr/bin/env python3
"""
Model 5: Embedding-constrained proportional covariance SEM.

Addresses the reviewer concern that Models 2-3 use freely estimated residual
covariances that can absorb any cross-scale variance, not just semantic overlap.

Model 5 fixes the 15 cross-scale residual covariances to be proportional to
the mean semantic similarity of each subscale pairing, with a single free
scaling parameter c optimized via profile likelihood. This makes the entire
cross-scale covariance structure embedding-determined.

If Model 5's latent correlation reduction is close to Models 2-3 (~25-29%),
the freely estimated covariances were indeed capturing semantic overlap.
If closer to Model 4 (~15%), the reviewer is right that Models 2-3 absorbed
non-semantic variance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')
import semopy

BASE = Path("/Users/devon7y/VS_Code/LLM_Factor_Analysis")

# ─── Load data ───────────────────────────────────────────────────────

tas_items = pd.read_csv(BASE / "scale_items/TAS-20_items.csv")
aq_items = pd.read_csv(BASE / "scale_items/AQ-50_items.csv")
tas_emb = np.load(BASE / "embeddings/TAS-20_items_8B.npz", allow_pickle=True)['embeddings']
aq_emb = np.load(BASE / "embeddings/AQ-50_items_8B.npz", allow_pickle=True)['embeddings']
tas_resp = pd.read_csv(BASE / "scale_responses/TAS-20_data.csv", sep='\t')
aq_resp = pd.read_csv(BASE / "scale_responses/AQ-50_data.csv", sep='\t')

n_tas, n_aq = len(tas_items), len(aq_items)
tas_factors = tas_items['factor'].values
aq_factors = aq_items['factor'].values

# Semantic similarity
combined_emb = np.vstack([tas_emb, aq_emb])
S_full = cosine_similarity(combined_emb)
S_cross = S_full[:n_tas, n_tas:]

# ─── Compute subscale scores ─────────────────────────────────────────

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
rename_map_raw = {
    'Attention_Switching': 'AtnSw',
    'Attention_to_Detail': 'AtnDt',
    'Communication': 'Comm',
    'Imagination': 'Imag',
    'Social_Skills': 'SocSk',
}
for k, v in aq_subscales.items():
    short = rename_map_raw.get(k, k.replace(' ', '_'))
    subscale_data[short] = v

sem_df = pd.DataFrame(subscale_data)
print(f"SEM data: {sem_df.shape}, columns: {sem_df.columns.tolist()}")

# ─── Compute mean semantic similarity per subscale pairing ───────────

aq_factor_to_short = rename_map_raw
tas_factor_names = sorted(tas_items['factor'].unique())
aq_factor_names = sorted(aq_items['factor'].unique())

mean_S_pairings = {}
for tf in tas_factor_names:
    for af in aq_factor_names:
        tas_mask = tas_factors == tf
        aq_mask = aq_factors == af
        mean_s = S_cross[np.ix_(tas_mask, aq_mask)].mean()
        af_short = aq_factor_to_short.get(af, af.replace(' ', '_'))
        mean_S_pairings[(tf, af_short)] = mean_s

print("\nMean semantic similarity per subscale pairing:")
for (tf, af), s in sorted(mean_S_pairings.items()):
    print(f"  {tf:>5} × {af:<6}: {s:.4f}")

grand_mean_S = np.mean(list(mean_S_pairings.values()))
print(f"\nGrand mean cross-scale S: {grand_mean_S:.4f}")

# Centered weights (for residual covariances, center so the latent
# factor absorbs the mean and residuals capture differential overlap)
centered_S = {k: v - grand_mean_S for k, v in mean_S_pairings.items()}
print("\nCentered pairing weights (mean-subtracted):")
for (tf, af), s in sorted(centered_S.items(), key=lambda x: -x[1]):
    print(f"  {tf:>5} × {af:<6}: {s:+.4f}")


# ─── Helper: extract latent correlation from semopy model ────────────

def get_latent_corr(model):
    est = model.inspect()
    tas_aq = est[(est['lval'] == 'TAS') & (est['rval'] == 'AQ') & (est['op'] == '~~')]
    if len(tas_aq) == 0:
        tas_aq = est[(est['lval'] == 'AQ') & (est['rval'] == 'TAS') & (est['op'] == '~~')]
    tas_var = est[(est['lval'] == 'TAS') & (est['rval'] == 'TAS') & (est['op'] == '~~')]
    aq_var = est[(est['lval'] == 'AQ') & (est['rval'] == 'AQ') & (est['op'] == '~~')]
    if len(tas_aq) == 0 or len(tas_var) == 0 or len(aq_var) == 0:
        return None
    cov = tas_aq['Estimate'].values[0]
    sd_t = np.sqrt(abs(tas_var['Estimate'].values[0]))
    sd_a = np.sqrt(abs(aq_var['Estimate'].values[0]))
    if sd_t == 0 or sd_a == 0:
        return None
    return cov / (sd_t * sd_a)


# ─── Model 1: Baseline (for reference) ──────────────────────────────

print("\n" + "=" * 70)
print("MODEL 1: BASELINE CFA")
print("=" * 70)

model1_spec = """
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
TAS ~~ AQ
"""
model1 = semopy.Model(model1_spec)
model1.fit(sem_df)
latent_r1 = get_latent_corr(model1)
stats1 = semopy.calc_stats(model1)
ll1 = stats1['ML'].values[0] if 'ML' in stats1.columns else None
print(f"Latent r = {latent_r1:.4f}")
print(f"CFI = {stats1['CFI'].values[0]:.3f}, RMSEA = {stats1['RMSEA'].values[0]:.3f}")


# ─── Model 5a: Proportional covariances (UNcentered) ────────────────
# θ_kl = c × mean_S_kl
# Profile likelihood over c

print("\n" + "=" * 70)
print("MODEL 5a: PROPORTIONAL COVARIANCES (UNCENTERED)")
print("=" * 70)
print("θ_kl = c × mean_S(TAS_k, AQ_l)")

def build_proportional_spec(c, weights, centered=False):
    lines = [
        "TAS =~ DIF + DDF + EOT",
        "AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag",
        "TAS ~~ AQ",
    ]
    for (tf, af), w in weights.items():
        val = c * w
        lines.append(f"{tf} ~~ {val:.8f}*{af}")
    return "\n".join(lines)

def fit_proportional_model(c, weights):
    spec = build_proportional_spec(c, weights)
    try:
        m = semopy.Model(spec)
        res = m.fit(sem_df)
        if hasattr(res, 'success') and not res.success:
            return None, None, None, None, None
        r = get_latent_corr(m)
        st = semopy.calc_stats(m)
        obj = float(res.fun) if hasattr(res, 'fun') else None
        chi2 = st['chi2'].values[0] if 'chi2' in st.columns else None
        return r, obj, chi2, m, st
    except Exception as e:
        return None, None, None, None, None

# Grid search over c for uncentered weights
print("\nGrid search over c (uncentered weights)...")
c_grid = np.linspace(-0.05, 0.15, 2000)
results_5a = []
for c in c_grid:
    r, obj, chi2, m, st = fit_proportional_model(c, mean_S_pairings)
    if r is not None and chi2 is not None:
        results_5a.append((c, r, chi2))

if results_5a:
    results_5a = np.array(results_5a)
    best_idx = np.argmin(results_5a[:, 2])
    c_opt, r_opt, chi2_opt = results_5a[best_idx]
    print(f"\nOptimal c = {c_opt:.5f}")
    print(f"Latent r at optimal c = {r_opt:.4f}")
    print(f"Chi² at optimal c = {chi2_opt:.1f}")
    if latent_r1:
        reduction_5a = (1 - r_opt / latent_r1) * 100
        print(f"Reduction from baseline: {reduction_5a:.1f}%")

    # Refit at optimal c for full stats
    _, _, _, m5a, st5a = fit_proportional_model(c_opt, mean_S_pairings)
    if st5a is not None:
        print(f"CFI = {st5a['CFI'].values[0]:.3f}, RMSEA = {st5a['RMSEA'].values[0]:.3f}")
        df5a = st5a['DoF'].values[0] if 'DoF' in st5a.columns else None
        print(f"df = {df5a}")

    # Print the fixed covariance values at optimal c
    print(f"\nFixed residual covariances at optimal c = {c_opt:.5f}:")
    for (tf, af), s in sorted(mean_S_pairings.items()):
        print(f"  {tf:>5} × {af:<6}: θ = {c_opt * s:.5f}  (S = {s:.4f})")
else:
    print("  All models failed to converge.")
    c_opt = None


# ─── Model 5b: Proportional covariances (CENTERED) ──────────────────
# θ_kl = c × (mean_S_kl - grand_mean_S)

print("\n" + "=" * 70)
print("MODEL 5b: PROPORTIONAL COVARIANCES (CENTERED)")
print("=" * 70)
print("θ_kl = c × (mean_S_kl - grand_mean)")

print("\nGrid search over c (centered weights)...")
c_grid_b = np.linspace(-0.2, 0.5, 2000)
results_5b = []
for c in c_grid_b:
    r, obj, chi2, m, st = fit_proportional_model(c, centered_S)
    if r is not None and chi2 is not None:
        results_5b.append((c, r, chi2))

if results_5b:
    results_5b = np.array(results_5b)
    best_idx_b = np.argmin(results_5b[:, 2])
    c_opt_b, r_opt_b, chi2_opt_b = results_5b[best_idx_b]
    print(f"\nOptimal c = {c_opt_b:.5f}")
    print(f"Latent r at optimal c = {r_opt_b:.4f}")
    print(f"Chi² at optimal c = {chi2_opt_b:.1f}")
    if latent_r1:
        reduction_5b = (1 - r_opt_b / latent_r1) * 100
        print(f"Reduction from baseline: {reduction_5b:.1f}%")

    _, _, _, m5b, st5b = fit_proportional_model(c_opt_b, centered_S)
    if st5b is not None:
        print(f"CFI = {st5b['CFI'].values[0]:.3f}, RMSEA = {st5b['RMSEA'].values[0]:.3f}")
        df5b = st5b['DoF'].values[0] if 'DoF' in st5b.columns else None
        print(f"df = {df5b}")

    print(f"\nFixed residual covariances at optimal c = {c_opt_b:.5f}:")
    for (tf, af), s in sorted(centered_S.items(), key=lambda x: -abs(x[1])):
        print(f"  {tf:>5} × {af:<6}: θ = {c_opt_b * s:+.5f}  (centered S = {s:+.4f})")
else:
    print("  All models failed to converge.")
    c_opt_b = None


# ─── Model 5c: DIF-only proportional covariances ────────────────────
# Only DIF pairings get residual covariances, proportional to S

print("\n" + "=" * 70)
print("MODEL 5c: DIF-ONLY PROPORTIONAL COVARIANCES")
print("=" * 70)
print("θ_kl = c × mean_S(DIF, AQ_l) for DIF pairings only")

dif_pairings = {k: v for k, v in mean_S_pairings.items() if k[0] == 'DIF'}
print("\nDIF pairing similarities:")
for (tf, af), s in sorted(dif_pairings.items()):
    print(f"  DIF × {af:<6}: S = {s:.4f}")

print("\nGrid search over c (DIF-only)...")
c_grid_c = np.linspace(-0.05, 0.20, 2000)
results_5c = []

def build_dif_only_spec(c, dif_weights):
    lines = [
        "TAS =~ DIF + DDF + EOT",
        "AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag",
        "TAS ~~ AQ",
    ]
    for (tf, af), w in dif_weights.items():
        val = c * w
        lines.append(f"{tf} ~~ {val:.8f}*{af}")
    return "\n".join(lines)

for c in c_grid_c:
    spec = build_dif_only_spec(c, dif_pairings)
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
        pass

if results_5c:
    results_5c = np.array(results_5c)
    best_idx_c = np.argmin(results_5c[:, 2])
    c_opt_c, r_opt_c, chi2_opt_c = results_5c[best_idx_c]

    def chi2_for_c(c_val):
        spec = build_dif_only_spec(c_val, dif_pairings)
        try:
            m = semopy.Model(spec)
            m.fit(sem_df)
            st = semopy.calc_stats(m)
            return st['chi2'].values[0]
        except:
            return 1e10

    from scipy.optimize import minimize_scalar
    refined = minimize_scalar(chi2_for_c, bounds=(c_opt_c - 0.005, c_opt_c + 0.005), method='bounded')
    if refined.success:
        c_opt_c = refined.x
        spec_ref = build_dif_only_spec(c_opt_c, dif_pairings)
        m_ref = semopy.Model(spec_ref)
        m_ref.fit(sem_df)
        r_opt_c = get_latent_corr(m_ref)
        st_ref = semopy.calc_stats(m_ref)
        chi2_opt_c = st_ref['chi2'].values[0]

    print(f"\nOptimal c = {c_opt_c:.5f}")
    print(f"Latent r at optimal c = {r_opt_c:.4f}")
    print(f"Chi² at optimal c = {chi2_opt_c:.1f}")
    if latent_r1:
        reduction_5c = (1 - r_opt_c / latent_r1) * 100
        print(f"Reduction from baseline: {reduction_5c:.1f}%")

    spec_opt = build_dif_only_spec(c_opt_c, dif_pairings)
    m5c = semopy.Model(spec_opt)
    m5c.fit(sem_df)
    st5c = semopy.calc_stats(m5c)
    print(f"CFI = {st5c['CFI'].values[0]:.3f}, RMSEA = {st5c['RMSEA'].values[0]:.3f}")
    df5c = st5c['DoF'].values[0] if 'DoF' in st5c.columns else None
    print(f"df = {df5c}")

    print(f"\nFixed DIF residual covariances at optimal c = {c_opt_c:.5f}:")
    for (tf, af), s in sorted(dif_pairings.items()):
        print(f"  DIF × {af:<6}: θ = {c_opt_c * s:.5f}  (S = {s:.4f})")

    # Also report the DIF covariance z-values from the free model for comparison
    print("\n  (Compare with freely estimated Model 2 DIF covariances)")
else:
    print("  All models failed to converge.")


# ─── Summary ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: MODEL COMPARISON")
print("=" * 70)

print(f"\n{'Model':<45} {'r':>6} {'Reduction':>10} {'df':>4} {'χ²':>8}")
print("-" * 78)
print(f"{'1. Baseline CFA':<45} {latent_r1:>6.3f} {'---':>10}")

# Refit Models 2-4 for comparison
model2_spec = """
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
TAS ~~ AQ
DIF ~~ Imag
DIF ~~ Comm
DIF ~~ AtnSw
DIF ~~ SocSk
"""
model2 = semopy.Model(model2_spec)
model2.fit(sem_df)
r2 = get_latent_corr(model2)
st2 = semopy.calc_stats(model2)
red2 = (1 - r2/latent_r1)*100 if r2 and latent_r1 else None
print(f"{'2. DIF residuals (4 free)':<45} {r2:>6.3f} {red2:>9.1f}% {st2['DoF'].values[0]:>4.0f} {st2['chi2'].values[0]:>8.1f}")

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
model3.fit(sem_df)
r3 = get_latent_corr(model3)
st3 = semopy.calc_stats(model3)
red3 = (1 - r3/latent_r1)*100 if r3 and latent_r1 else None
print(f"{'3. All residuals (15 free)':<45} {r3:>6.3f} {red3:>9.1f}% {st3['DoF'].values[0]:>4.0f} {st3['chi2'].values[0]:>8.1f}")

# Model 4: semantic method factor
all_weights = {}
for tf in tas_factor_names:
    tas_mask = tas_factors == tf
    all_weights[tf] = S_cross[tas_mask, :].mean()
for af in aq_factor_names:
    aq_mask = aq_factors == af
    af_short = aq_factor_to_short.get(af, af.replace(' ', '_'))
    all_weights[af_short] = S_cross[:, aq_mask].mean()
mean_w = np.mean(list(all_weights.values()))
sd_w = np.std(list(all_weights.values()))
norm_w = {k: (v - mean_w)/sd_w for k, v in all_weights.items()}
loading_strs = [f"{w:.4f}*{var}" for var, w in norm_w.items()]
model4_spec = f"""
TAS =~ DIF + DDF + EOT
AQ =~ SocSk + Comm + AtnSw + AtnDt + Imag
Sem =~ {' + '.join(loading_strs)}
TAS ~~ AQ
Sem ~~ TAS
Sem ~~ AQ
"""
model4 = semopy.Model(model4_spec)
model4.fit(sem_df)
r4 = get_latent_corr(model4)
st4 = semopy.calc_stats(model4)
red4 = (1 - r4/latent_r1)*100 if r4 and latent_r1 else None
print(f"{'4. Semantic method factor (uniform loadings)':<45} {r4:>6.3f} {red4:>9.1f}% {st4['DoF'].values[0]:>4.0f} {st4['chi2'].values[0]:>8.1f}")

# Model 5a
if results_5a is not None and len(results_5a) > 0:
    print(f"{'5a. Proportional covs, all 15 (1 free c)':<45} {r_opt:>6.3f} {reduction_5a:>9.1f}% {df5a:>4.0f} {chi2_opt:>8.1f}")

# Model 5b
if results_5b is not None and len(results_5b) > 0:
    print(f"{'5b. Centered proportional covs (1 free c)':<45} {r_opt_b:>6.3f} {reduction_5b:>9.1f}% {df5b:>4.0f} {chi2_opt_b:>8.1f}")

# Model 5c
if results_5c is not None and len(results_5c) > 0:
    print(f"{'5c. DIF-only proportional covs (1 free c)':<45} {r_opt_c:>6.3f} {reduction_5c:>9.1f}% {df5c:>4.0f} {chi2_opt_c:>8.1f}")

print(f"\n{'Observed-score partial correlation':<45} {'---':>6} {'33.4%':>10}")
print(f"{'  (raw r = .314, partial r = .209)':<45}")

print("\nKey question: Does embedding-constrained Model 5 achieve similar")
print("reduction to freely-estimated Models 2-3? If yes, the reduction")
print("is attributable to semantic overlap. If closer to Model 4's 15%,")
print("Models 2-3 absorbed non-semantic variance.")
