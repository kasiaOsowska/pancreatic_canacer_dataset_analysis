"""
3a. Stability check for CovariatesResidualTransformer (age slope per gene).

Cel: sprawdzic, czy per-gen wspolczynniki regresji  expression ~ age  liczone
na zdrowych probkach (uzywane przez CovariatesResidualTransformer / AgeBiasReductor)
sa stabilne miedzy losowymi podsplitami train. Jesli nie -- residua liczone
przez ten transformer nie sa wiarygodne.

Schemat:
0. Holdout split (jak w glownym skrypcie); pracujemy tylko na train.
1. N_ITER losowych subsplitow train. Dla kazdego: fit pipeline-do-AgeBiasReductor
   wlacznie i wyciagam coef_ (slope β) oraz intercept_ (α) z ostatniego stepu.
2. Per gen agreguje rozklad β po iteracjach -> mean, std, CV, SNR (mean/std).
3. Bootstrap test H0: β = 0 i bootstrap CI dla mean β.
4. Split-half reliability: dziele iteracje na 2 polowki, licze mean β per gen
   w kazdej polowce, korelacja Pearsona/Spearmana po genach.
5. Wykresy diagnostyczne + CSV.

Interpretacja:
- CV(β) < 0.1  -> bardzo stabilne
- CV(β) > 0.5  -> niestabilne, residua niewiarygodne dla tego genu
- |SNR| > 2    -> β rzetelnie != 0 (nie szum)
- Split-half r > 0.9 -> wspolczynniki transferuja sie miedzy podzialami
- Split-half r < 0.5 -> wspolczynniki to glownie szum probkowania
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY
from utilz.preprocessing_utilz import (
    ConstantExpressionReductor,
    CovariatesResidualTransformer, CovariatesBiasReductor,
)

# ── paths ────────────────────────────────────────────────────────
meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

# ── hyperparameters ──────────────────────────────────────────────
N_ITER             = 50
HOLDOUT_TEST_SIZE  = 0.2
INNER_TEST_SIZE    = 0.4     # zostaje 60% train do fitu
CI_LEVEL           = 0.95
N_BOOTSTRAP        = 2000
BASE_SEED          = 2137
TOP_K_PLOT         = 30      # ile najbardziej niestabilnych genow w boxplocie
CV_STABLE          = 0.1
CV_UNSTABLE        = 0.5


def build_pipeline_until_age(ds, y_encoded):
    """Pipeline do AgeBiasReductor wlacznie -- zeby wyciagnac coef_ i intercept_."""
    sex_numeric = ds.sex.map({"F": 0, "M": 1})
    return Pipeline([
        ('ConstantExpressionReductor', ConstantExpressionReductor()),
        ('SexBiasReductor',            CovariatesBiasReductor(covariate=sex_numeric)),
        ('AgeBiasReductor',            CovariatesResidualTransformer(covariate=ds.age, labels=y_encoded)),
    ])


def run_iteration(ds, y_encoded, X_train_raw, y_train, iteration_idx, random_state):
    X_tr_raw, _, y_tr, _ = ds.get_train_test_valid_split(
        X_train_raw, y_train, test_size=INNER_TEST_SIZE, valid_size=0,
        random_state=random_state, return_valid=False,
    )
    pipe = build_pipeline_until_age(ds, y_encoded)
    pipe.fit(X_tr_raw, y_tr)

    age_step = pipe.named_steps['AgeBiasReductor']
    feature_names = list(age_step.selected_genes_)
    coef = np.asarray(age_step.coef_, dtype=float).ravel()
    intercept = np.asarray(age_step.intercept_, dtype=float).ravel()
    n_healthy = int((y_tr == 0).sum())

    print(f"[iter {iteration_idx+1:>3}/{N_ITER}]  inner_tr={len(X_tr_raw)}  "
          f"healthy={n_healthy}  feat={len(feature_names)}")
    return feature_names, coef, intercept


def per_gene_bootstrap(values, rng, n_boot=N_BOOTSTRAP, level=CI_LEVEL):
    """Bootstrap mean: dwustronny p-value dla H0: mean=0 + CI percentylowe."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return dict(boot_p=np.nan, boot_lo=np.nan, boot_hi=np.nan)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = v[rng.integers(0, n, size=n)].mean()
    p_lt = (boot <= 0).mean()
    p_gt = (boot >= 0).mean()
    p_two = float(2.0 * min(p_lt, p_gt))
    alpha = 1.0 - level
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return dict(boot_p=p_two, boot_lo=float(lo), boot_hi=float(hi))


def split_half_reliability(coef_matrix, gene_names, rng, n_repeats=20):
    """
    coef_matrix: (N_ITER, n_genes), NaN dla iteracji w ktorych genu nie bylo.
    Powtarzamy n_repeats razy losowy podzial iteracji na 2 polowki, liczymy
    mean β per gen w kazdej polowce (z pominieciem NaN), liczymy korelacje
    Pearsona i Spearmana po genach. Zwracamy mediane korelacji.
    """
    n_iter = coef_matrix.shape[0]
    pearson_rs, spearman_rs = [], []
    for _ in range(n_repeats):
        order = rng.permutation(n_iter)
        a = order[:n_iter//2]
        b = order[n_iter//2:]
        mean_a = np.nanmean(coef_matrix[a, :], axis=0)
        mean_b = np.nanmean(coef_matrix[b, :], axis=0)
        mask = np.isfinite(mean_a) & np.isfinite(mean_b)
        if mask.sum() < 5:
            continue
        rp, _ = stats.pearsonr(mean_a[mask], mean_b[mask])
        rs, _ = stats.spearmanr(mean_a[mask], mean_b[mask])
        pearson_rs.append(rp)
        spearman_rs.append(rs)
    return float(np.median(pearson_rs)), float(np.median(spearman_rs)), \
           float(np.std(pearson_rs)), float(np.std(spearman_rs))


def main():
    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

    print("Class mapping:")
    for cls in le.classes_:
        print(f"  {cls} -> {le.transform([cls])[0]}")

    # ── Holdout (zgodnie z glownym skryptem) ────────────────────
    X_train_raw, _, y_train, _ = ds.get_train_test_valid_split(
        ds.X, y_encoded, test_size=HOLDOUT_TEST_SIZE, valid_size=0,
        random_state=BASE_SEED, return_valid=False,
    )
    print(f"\nTrain (po holdout): n={len(X_train_raw)}  "
          f"(healthy={int((y_train==0).sum())}, disease={int((y_train==1).sum())})")

    # ── per-gen lista β i α ─────────────────────────────────────
    iter_results = []  # [(feat, coef, intercept) per iteracja]
    all_genes = set()
    for i in range(N_ITER):
        rs = BASE_SEED + i + 1
        feat, coef, intercept = run_iteration(
            ds, y_encoded, X_train_raw, y_train, i, random_state=rs)
        iter_results.append((feat, coef, intercept))
        all_genes.update(feat)

    # ── coef matrix (N_ITER x n_genes), NaN gdzie genu nie bylo ─
    gene_list = sorted(all_genes)
    gene_to_col = {g: j for j, g in enumerate(gene_list)}
    coef_matrix = np.full((N_ITER, len(gene_list)), np.nan)
    intercept_matrix = np.full((N_ITER, len(gene_list)), np.nan)
    for i, (feat, coef, intercept) in enumerate(iter_results):
        for j, g in enumerate(feat):
            col = gene_to_col[g]
            coef_matrix[i, col] = coef[j]
            intercept_matrix[i, col] = intercept[j]

    # ── statystyki per gen ──────────────────────────────────────
    rng = np.random.default_rng(BASE_SEED)
    records = []
    for j, gene in enumerate(gene_list):
        b = coef_matrix[:, j]
        b = b[np.isfinite(b)]
        a = intercept_matrix[:, j]
        a = a[np.isfinite(a)]
        if len(b) < 2:
            continue
        mean_b = float(b.mean())
        std_b = float(b.std(ddof=1))
        cv_b = float(abs(std_b / mean_b)) if abs(mean_b) > 1e-12 else float('inf')
        snr_b = float(mean_b / std_b) if std_b > 1e-12 else float('inf')
        bs = per_gene_bootstrap(b, rng)
        records.append({
            'gene': gene,
            'n_iter_seen': int(len(b)),
            'mean_coef': mean_b,
            'std_coef': std_b,
            'cv_coef': cv_b,
            'snr_coef': snr_b,
            'boot_p_value': bs['boot_p'],
            'boot_ci_lo': bs['boot_lo'],
            'boot_ci_hi': bs['boot_hi'],
            'mean_intercept': float(a.mean()),
            'std_intercept': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        })

    coef_df = (
        pd.DataFrame(records)
        .sort_values('cv_coef', ascending=False)
        .reset_index(drop=True)
    )
    coef_df.index += 1
    coef_df.index.name = 'rank_by_cv'
    coef_df.to_csv('covariates_age_coef_stability.csv', index=True)

    # ── globalne podsumowanie ───────────────────────────────────
    n = len(coef_df)
    cv_finite = coef_df['cv_coef'].replace([np.inf, -np.inf], np.nan).dropna()
    snr_finite = coef_df['snr_coef'].abs().replace([np.inf, -np.inf], np.nan).dropna()
    n_reliable = int((coef_df['boot_p_value'] < 0.05).sum())
    n_high_snr = int((coef_df['snr_coef'].abs() > 2).sum())
    n_unstable = int((coef_df['cv_coef'] > CV_UNSTABLE).sum())
    n_stable = int((coef_df['cv_coef'] < CV_STABLE).sum())

    print("\n" + "="*64)
    print("PODSUMOWANIE STABILNOSCI WSPOLCZYNNIKA β (slope wzgledem age)")
    print("="*64)
    print(f"  Geny ogolem:                                {n}")
    print(f"  β rzetelnie != 0 (boot p<0.05):             {n_reliable:>5} ({100*n_reliable/n:.1f}%)")
    print(f"  |SNR| > 2 (mean/std β):                     {n_high_snr:>5} ({100*n_high_snr/n:.1f}%)")
    print(f"  CV < {CV_STABLE} (bardzo stabilne):                  {n_stable:>5} ({100*n_stable/n:.1f}%)")
    print(f"  CV > {CV_UNSTABLE} (niestabilne, residua suspect):    {n_unstable:>5} ({100*n_unstable/n:.1f}%)")
    if len(cv_finite):
        print(f"  Mediana CV(β):                              {cv_finite.median():.3f}")
        print(f"  Q75 CV(β):                                  {cv_finite.quantile(0.75):.3f}")
    if len(snr_finite):
        print(f"  Mediana |SNR(β)|:                           {snr_finite.median():.3f}")

    # ── split-half reliability ──────────────────────────────────
    rp_med, rs_med, rp_sd, rs_sd = split_half_reliability(
        coef_matrix, gene_list, rng, n_repeats=30,
    )
    print(f"\nSplit-half reliability mean β (mediana z 30 losowych podzialow iteracji):")
    print(f"  Pearson  r = {rp_med:.4f}  (sd = {rp_sd:.4f})")
    print(f"  Spearman ρ = {rs_med:.4f}  (sd = {rs_sd:.4f})")
    print(f"  Interpretacja: r > 0.9 = bardzo stabilne; r < 0.5 = w duzej mierze szum.")

    # ── przyklady najbardziej / najmniej stabilnych genow ───────
    print(f"\nTop 15 najbardziej NIESTABILNYCH genow (po CV):")
    print(coef_df.head(15)[['gene','n_iter_seen','mean_coef','std_coef','cv_coef',
                            'snr_coef','boot_p_value']].to_string())
    print(f"\nTop 15 najbardziej STABILNYCH genow (najnizsze CV, β faktycznie != 0):")
    stable = coef_df[coef_df['boot_p_value'] < 0.05].sort_values('cv_coef').head(15)
    print(stable[['gene','n_iter_seen','mean_coef','std_coef','cv_coef',
                  'snr_coef','boot_p_value']].to_string())

    # ── wykresy diagnostyczne ───────────────────────────────────
    plot_diagnostics(coef_df, coef_matrix, gene_list, gene_to_col)


def plot_diagnostics(coef_df, coef_matrix, gene_list, gene_to_col):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) histogram CV
    cv = coef_df['cv_coef'].replace([np.inf, -np.inf], np.nan).dropna()
    cv_clip = cv.clip(upper=3)
    axes[0,0].hist(cv_clip, bins=60, color='#6366f1', alpha=0.85, edgecolor='white')
    axes[0,0].axvline(CV_STABLE, color='#10b981', ls='--',
                      label=f'CV = {CV_STABLE} (stabilne)')
    axes[0,0].axvline(CV_UNSTABLE, color='#ef4444', ls='--',
                      label=f'CV = {CV_UNSTABLE} (niestabilne)')
    axes[0,0].set_xlabel('CV(β) = std/|mean|  (clipped at 3)')
    axes[0,0].set_ylabel('liczba genow')
    axes[0,0].set_title('Rozklad CV wspolczynnika β  (lower = better)')
    axes[0,0].legend()

    # (b) scatter |mean β| vs std β  (signal vs noise)
    mean_abs = coef_df['mean_coef'].abs().values
    std_v = coef_df['std_coef'].values
    axes[0,1].scatter(mean_abs, std_v, s=4, alpha=0.4, color='#6366f1')
    mx = float(np.nanpercentile(np.concatenate([mean_abs, std_v]), 99))
    xs = np.linspace(0, mx, 100)
    axes[0,1].plot(xs, xs,    'k--', alpha=0.5, label='SNR = 1 (β = noise)')
    axes[0,1].plot(xs, xs/2,  'r--', alpha=0.5, label='SNR = 2 (β reliable)')
    axes[0,1].set_xlim(0, mx); axes[0,1].set_ylim(0, mx)
    axes[0,1].set_xlabel('|mean β|')
    axes[0,1].set_ylabel('std β  (po iteracjach)')
    axes[0,1].set_title('Signal-to-noise per gene')
    axes[0,1].legend()

    # (c) top 30 najbardziej niestabilnych genow: boxplot β
    top = coef_df.head(min(TOP_K_PLOT, len(coef_df)))
    box_data = []
    box_labels = []
    for g in top['gene']:
        col = gene_to_col[g]
        v = coef_matrix[:, col]
        v = v[np.isfinite(v)]
        if len(v) > 0:
            box_data.append(v)
            box_labels.append(g)
    axes[1,0].boxplot(box_data, vert=False, tick_labels=box_labels, showmeans=True)
    axes[1,0].axvline(0, color='red', ls='--', lw=0.8)
    axes[1,0].invert_yaxis()
    axes[1,0].set_xlabel('β  (across iterations)')
    axes[1,0].set_title(f'Top {len(box_labels)} najmniej stabilnych genow (po CV)')
    axes[1,0].tick_params(axis='y', labelsize=7)

    # (d) histogram bootstrap p-values
    pvals = coef_df['boot_p_value'].dropna().values
    axes[1,1].hist(pvals, bins=40, color='#f59e0b', alpha=0.85, edgecolor='white')
    axes[1,1].axvline(0.05, color='#ef4444', ls='--', label='p = 0.05')
    axes[1,1].set_xlabel('Bootstrap p-value  (H0: mean β = 0)')
    axes[1,1].set_ylabel('liczba genow')
    axes[1,1].set_title('Czy slope β jest istotnie != 0?')
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig('covariates_age_coef_stability.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()
