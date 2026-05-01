"""
3. Random Resampling Gene Stability Analysis (ANOVA F-only, simplified)

Schemat:
0. Holdout: jednorazowy podzial train/test -- test NIE jest dotykany podczas selekcji
1. W kazdej iteracji (N_ITER): losowy subsplit *tylko w obrebie train* (stratified)
2. Score per gen w iteracji: statystyka F z one-way ANOVA (model-free)
3. Dla kazdego genu: rozklad empiryczny po iteracjach + CI (percentylowe + bootstrap)
4. Finalna walidacja top-K genow na czystym holdout test
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import f_classif, SelectKBest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY
from utilz.preprocessing_utilz import (
    ConstantExpressionReductor, AnovaReductor,
    MeanExpressionReductor,
    WithinGroupVarianceReductor, AnovaFdrReductor,
    CovariatesResidualTransformer, CovariatesBiasReductor,
)

# ── paths ────────────────────────────────────────────────────────
meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

# ── hyperparameters ──────────────────────────────────────────────
N_ITER              = 50      # liczba losowych resamplowan (wewnatrz train)
HOLDOUT_TEST_SIZE   = 0.2     # jednorazowy holdout (czysty test na koniec)
INNER_TEST_SIZE     = 0.1     # rozmiar "wewnetrznego" val w kazdej iteracji
TOP_K_PER_ITER      = 5000    # ile top genow zapisywac z kazdej iteracji
TOP_K_FINAL         = 12      # ile top genow walidowac na holdout test
ANOVA_PERCENTILE    = 5
MEAN_PERCENTILE     = 5
WITHIN_GROUP_VAR_P  = 95
ANOVA_FDR_THRESHOLD = 0.1
CI_LEVEL            = 0.95    # przedzial ufnosci (np. 0.95 -> 2.5%/97.5%)
N_BOOTSTRAP         = 2000    # liczba bootstrap-resamplowan dla CI sredniej
BASE_SEED           = 2137


def build_pipeline(ds, y_encoded):
    sex_numeric = ds.sex.map({"F": 0, "M": 1})
    return Pipeline([
        ('ConstantExpressionReductor', ConstantExpressionReductor()),
        ('SexBiasReductor',             CovariatesBiasReductor(covariate=sex_numeric)),
        #('AgeBiasReductor',            CovariatesResidualTransformer(covariate=ds.age, labels=y_encoded)),
        ('scaler',                     StandardScaler()),
    ])


def run_iteration(ds, y_encoded, X_train_raw, y_train, iteration_idx, random_state):

    X_tr_raw, X_te_raw, y_tr, y_te = ds.get_train_test_valid_split(
        X_train_raw, y_train, test_size=INNER_TEST_SIZE, valid_size=0,
        random_state=random_state, return_valid=False,
    )

    pipe = build_pipeline(ds, y_encoded)
    X_tr = pipe.fit_transform(X_tr_raw, y_tr)
    #X_te = pipe.transform(X_te_raw)
    feature_names = list(pipe.named_steps['ConstantExpressionReductor'].selected_genes_)
    k = min(TOP_K_PER_ITER, X_tr.shape[1])
    skb = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
    F = np.nan_to_num(skb.scores_, nan=0.0, posinf=0.0, neginf=0.0)

    gene_iter_score = F

    print(f"[iter {iteration_idx+1:>3}/{N_ITER}]  "
          f"inner_tr={len(X_tr_raw)}  inner_val={len(X_te_raw)}  "
          f"feat={len(feature_names)}  method=anova_f")

    return {
        'feature_names': feature_names,
        'gene_iter_score': gene_iter_score,
    }


def compute_ci(values, level=CI_LEVEL, n_boot=N_BOOTSTRAP, rng=None):
    """Zwraca CI percentylowe (empirical distribution) oraz bootstrap-CI dla sredniej."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return dict(mean=np.nan, std=np.nan, median=np.nan,
                    pct_lo=np.nan, pct_hi=np.nan,
                    boot_mean_lo=np.nan, boot_mean_hi=np.nan)

    alpha = 1.0 - level
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    pct_lo, pct_hi = np.percentile(values, [lo_q, hi_q])

    if rng is None:
        rng = np.random.default_rng(BASE_SEED)
    boot_means = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    boot_lo, boot_hi = np.percentile(boot_means, [lo_q, hi_q])

    return dict(
        mean=values.mean(),
        std=values.std(ddof=1) if len(values) > 1 else 0.0,
        median=np.median(values),
        pct_lo=pct_lo,
        pct_hi=pct_hi,
        boot_mean_lo=boot_lo,
        boot_mean_hi=boot_hi,
    )


def main():
    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

    print("Class mapping:")
    for cls in le.classes_:
        print(f"  {cls} -> {le.transform([cls])[0]}")
    print("\nClass distribution:")
    print(y_encoded.value_counts().sort_index())

    # ── 0. Holdout train/test split (czysty test na sam koniec) ─
    X_train_raw, X_test_raw, y_train, y_test = ds.get_train_test_valid_split(
        ds.X, y_encoded, test_size=HOLDOUT_TEST_SIZE, valid_size=0,
        random_state=BASE_SEED, return_valid=False,
    )
    print(f"\nHoldout split:")
    print(f"  Train: {len(X_train_raw)}  (cancer={int(y_train.sum())}, "
          f"ctrl={int((y_train==0).sum())})")
    print(f"  Test:  {len(X_test_raw)}   (cancer={int(y_test.sum())},  "
          f"ctrl={int((y_test==0).sum())})  <- NIE dotykany podczas selekcji")

    # ── per-gene: lista wartosci score z kolejnych iteracji ─────
    gene_scores_all = defaultdict(list)

    for i in range(N_ITER):
        rs = BASE_SEED + i + 1  # +1 zeby nie pokryc sie z holdout seed
        res = run_iteration(ds, y_encoded, X_train_raw, y_train, i, random_state=rs)

        feat = res['feature_names']
        combined = np.asarray(res['gene_iter_score'], dtype=float)

        # tylko top-K genow z tej iteracji (wg statystyki F)
        k = min(TOP_K_PER_ITER, len(feat))
        top_idx = np.argsort(combined)[::-1][:k]
        top_mask = np.zeros(len(feat), dtype=bool)
        top_mask[top_idx] = True

        for j, gene in enumerate(feat):
            if top_mask[j]:
                gene_scores_all[gene].append(float(combined[j]))

    # ── rozklad empiryczny + CI per gen ─────────────────────────
    rng = np.random.default_rng(BASE_SEED)
    records = []
    for gene, vals in gene_scores_all.items():
        ci = compute_ci(vals, level=CI_LEVEL, n_boot=N_BOOTSTRAP, rng=rng)
        records.append({
            'gene': gene,
            'score': (len(vals)/N_ITER) * ci['boot_mean_ci_lo'],
            'n_iter_seen': len(vals),
            'mean_score': ci['mean'],
            'median_score': ci['median'],
            'std_score': ci['std'],
            'emp_ci_lo': ci['pct_lo'],
            'emp_ci_hi': ci['pct_hi'],
            'boot_mean_ci_lo': ci['boot_mean_lo'],
            'boot_mean_ci_hi': ci['boot_mean_hi'],
        })

    stability_df = (
        pd.DataFrame(records)
        .sort_values('score', ascending=False)
        .reset_index(drop=True)
    )
    stability_df.index += 1
    stability_df.index.name = 'rank'
    # save
    stability_df.to_csv('gene_stability_summary.csv', index=True)

    print(f"\nTotal unique genes seen: {len(stability_df)}")
    print("\nTop 30 genow (sort: score = freq * mean_F):")
    print(stability_df.head(30).to_string())

    print(f"\n{'='*60}")
    print(f"FINALNA WALIDACJA na holdout test (n={len(X_test_raw)})")
    print('='*60)

    pipe_full = build_pipeline(ds, y_encoded)
    X_tr_full = pipe_full.fit_transform(X_train_raw, y_train)
    X_te_full = pipe_full.transform(X_test_raw)
    all_feat = list(pipe_full.named_steps['SexBiasReductor'].selected_genes_)

    X_tr_df = pd.DataFrame(X_tr_full, columns=all_feat)
    X_te_df = pd.DataFrame(X_te_full, columns=all_feat)

    # ── INCREMENTAL TOP-K: wykres AUC w funkcji liczby genow ────
    # Bierzemy top 30 genow ze stability_df (ranking po `score`)
    # i dodajemy je po jednym: k=1, 2, ..., 30. Dla kazdego k
    # trenujemy 3 modele i zapisujemy AUC na holdout test.
    N_TOP = 30
    ranked_genes = stability_df['gene'].head(N_TOP).tolist()
    # uwzgledniamy tylko geny obecne w pelnym pipeline (po preprocessing)
    ranked_genes = [g for g in ranked_genes if g in X_tr_df.columns]
    n_steps = len(ranked_genes)
    print(f"Incremental ranking: top {n_steps} genow ze stability_df.")

    spw = Counter(y_train)[0] / Counter(y_train)[1]

    def make_models():
        return [
            ('LogReg', LogisticRegression(solver='saga', max_iter=15000,
                                          class_weight='balanced', fit_intercept=True)),
            ('XGB',    XGBClassifier(scale_pos_weight=spw, n_estimators=500,
                                     random_state=BASE_SEED, verbosity=0)),
            ('LGBM',   LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                      random_state=42, verbose=-1)),
        ]

    incremental_records = []
    for k in range(1, n_steps + 1):
        genes_k = ranked_genes[:k]
        added_gene = ranked_genes[k - 1]

        X_tr_sel = X_tr_df[genes_k].values
        X_te_sel = X_te_df[genes_k].values

        row = {'k': k, 'gene_added': added_gene}
        aucs_k = []
        for name, model in make_models():
            model.fit(X_tr_sel, y_train)
            auc = roc_auc_score(y_test, model.predict_proba(X_te_sel)[:, 1])
            row[name] = auc
            aucs_k.append(auc)
        row['MEAN'] = float(np.mean(aucs_k))

        incremental_records.append(row)
        print(f"  k={k:>2}  + {added_gene:20s}  "
              f"LogReg={row['LogReg']:.4f}  XGB={row['XGB']:.4f}  "
              f"LGBM={row['LGBM']:.4f}  MEAN={row['MEAN']:.4f}")

    incremental_df = pd.DataFrame(incremental_records)
    incremental_df.to_csv('gene_incremental_aucs.csv', index=False)

    # ── wykres: AUC vs k ────────────────────────────────────────
    fig_inc, ax_inc = plt.subplots(figsize=(10, 6))
    ks = incremental_df['k'].values
    for col, color in [('LogReg', '#6366f1'), ('XGB', '#10b981'),
                       ('LGBM', '#f59e0b'), ('MEAN', '#1f2937')]:
        lw = 2.5 if col == 'MEAN' else 1.5
        ls = '--' if col == 'MEAN' else '-'
        ax_inc.plot(ks, incremental_df[col].values,
                    marker='o', label=col, color=color, lw=lw, ls=ls)
    ax_inc.set_xlabel('Liczba genow (top-k wg stability ranking)')
    ax_inc.set_ylabel('AUC na holdout test')
    ax_inc.set_title(f'AUC vs liczba genow (incremental top-k z rankingu stability)')
    ax_inc.set_xticks(ks)
    ax_inc.grid(alpha=0.3)
    ax_inc.legend()
    plt.tight_layout()
    plt.savefig('gene_incremental_aucs.png', dpi=120)
    plt.show()


    # ── wykresy ─────────────────────────────────────────────────
    top30 = stability_df.head(30)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # (a) top30: mean z CI percentylowym
    y_pos = np.arange(len(top30))
    xerr_lo = top30['mean_score'] - top30['emp_ci_lo']
    xerr_hi = top30['emp_ci_hi'] - top30['mean_score']
    axes[0].barh(y_pos, top30['mean_score'],
                 xerr=[xerr_lo, xerr_hi], color='#6366f1',
                 alpha=0.85, edgecolor='white',
                 error_kw=dict(ecolor='#1f2937', lw=1.2, capsize=3))
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(
        [f"{g}  (n={int(n)})" for g, n in zip(top30['gene'], top30['n_iter_seen'])],
        fontsize=8,
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f'ANOVA F  -- mean +/- {int(CI_LEVEL*100)}% empirical CI')
    axes[0].set_title(f'Top 30 genow (N_ITER={N_ITER})')

    # (b) rozklad empiryczny top-10 (boxplot)
    top10_genes = top30['gene'].head(10).tolist()
    data_box = [gene_scores_all[g] for g in top10_genes]
    axes[1].boxplot(data_box, vert=False, tick_labels=top10_genes, showmeans=True)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('ANOVA F  (rozklad empiryczny)')
    axes[1].set_title('Top 10 -- rozklad po iteracjach')

    plt.tight_layout()
    plt.savefig('gene_stability_random_resampling.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()
