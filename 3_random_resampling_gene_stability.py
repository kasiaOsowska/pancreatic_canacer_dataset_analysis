"""
3. Random Resampling Gene Stability Analysis

Schemat:
0. Holdout: jednorazowy podzial train/test -- test NIE jest dotykany podczas selekcji
1. W kazdej iteracji (N_ITER): losowy subsplit *tylko w obrebie train* (stratified)
2. Trening 3 modeli (LogReg, XGB, LGBM) + SHAP na wewnetrznym zbiorze walidacyjnym
3. Agregacja per gen:
       score_i,model = |mean(SHAP_i)| / std(SHAP_i)   (SNR)
       gene_score    = sum_model(SHAP_score * AUC_model)
4. Dla kazdego genu: rozklad empiryczny po iteracjach + CI (percentylowe + bootstrap)
5. Finalna walidacja top-K genow na czystym holdout test
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import shap
from collections import Counter, defaultdict
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
N_ITER              = 20      # liczba losowych resamplowan (wewnatrz train)
HOLDOUT_TEST_SIZE   = 0.2     # jednorazowy holdout (czysty test na koniec)
INNER_TEST_SIZE     = 0.2     # rozmiar "wewnetrznego" val w kazdej iteracji
TOP_K_PER_ITER      = 100     # ile top genow zapisywac z kazdej iteracji
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
        ('AnovaFDRReductor',           AnovaFdrReductor(alpha=ANOVA_FDR_THRESHOLD)),
        #('AnovaReductor',              AnovaReductor(percentile=ANOVA_PERCENTILE)),
        ('WithinGroupVarianceReductor',WithinGroupVarianceReductor(WITHIN_GROUP_VAR_P)),
        ('MeanExpressionReductor',     MeanExpressionReductor(percentile=MEAN_PERCENTILE)),
        ('SexBiasReductor', CovariatesBiasReductor(covariate=sex_numeric)),
        ('AgeBiasReductor',            CovariatesResidualTransformer(covariate=ds.age, labels=y_encoded)),
        ('scaler',                     StandardScaler()),
    ])


def snr(shap_matrix):
    """|mean(SHAP)| / std(SHAP) per gene."""
    eps = 1e-10
    return np.abs(shap_matrix.mean(axis=0)) / (shap_matrix.std(axis=0) + eps)


def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


def run_iteration(ds, y_encoded, X_train_raw, y_train, iteration_idx, random_state):

    X_tr_raw, X_te_raw, y_tr, y_te = ds.get_train_test_valid_split(
        X_train_raw, y_train, test_size=INNER_TEST_SIZE, valid_size=0,
        random_state=random_state, return_valid=False,
    )

    pipe = build_pipeline(ds, y_encoded)
    X_tr = pipe.fit_transform(X_tr_raw, y_tr)
    X_te = pipe.transform(X_te_raw)
    feature_names = list(pipe.named_steps['SexBiasReductor'].selected_genes_)

    scale_pos_weight = Counter(y_tr)[0] / Counter(y_tr)[1]
    logreg = LogisticRegression(solver='saga', max_iter=15000,
                                class_weight='balanced', fit_intercept=True,
                                l1_ratio=0.2, random_state=BASE_SEED)
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=500,
                        random_state=BASE_SEED, verbosity=0)
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                          random_state=42, verbose=-1)

    for m in [logreg, xgb, lgbm]:
        m.fit(X_tr, y_tr)

    auc_lr = roc_auc_score(y_te, logreg.predict_proba(X_te)[:, 1])
    auc_xgb = roc_auc_score(y_te, xgb.predict_proba(X_te)[:, 1])
    auc_lgbm = roc_auc_score(y_te, lgbm.predict_proba(X_te)[:, 1])

    # SHAP per model
    sv_lr = shap.LinearExplainer(logreg, X_tr).shap_values(X_te)
    sv_lgbm = shap.TreeExplainer(lgbm).shap_values(X_te)
    sv_xgb = shap.TreeExplainer(xgb).shap_values(X_te)

    score_lr = snr(sv_lr)
    score_xgb = snr(sv_xgb)
    score_lgbm = snr(sv_lgbm)

    # normalization
    n_lr = minmax_norm(score_lr)
    n_xgb = minmax_norm(score_xgb)
    n_lgbm = minmax_norm(score_lgbm)

    gene_iter_score = n_lr * auc_lr + n_xgb * auc_xgb + n_lgbm * auc_lgbm

    print(f"[iter {iteration_idx+1:>3}/{N_ITER}]  "
          f"inner_tr={len(X_tr_raw)}  inner_val={len(X_te_raw)}  "
          f"feat={len(feature_names)}  "
          f"AUC lr={auc_lr:.4f} xgb={auc_xgb:.4f} lgbm={auc_lgbm:.4f}")

    return {
        'feature_names': feature_names,
        'gene_iter_score': gene_iter_score,
        'aucs': {'logreg': auc_lr, 'xgb': auc_xgb, 'lgbm': auc_lgbm},
        'per_model_scores': {
            'logreg': n_lr, 'xgb': n_xgb, 'lgbm': n_lgbm,
        },
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
    gene_scores_all = defaultdict(list)          # sum_model(SHAP*AUC)
    gene_scores_per_model = {                     # oddzielnie per model
        'logreg': defaultdict(list),
        'xgb':    defaultdict(list),
        'lgbm':   defaultdict(list),
    }
    auc_history = []   # AUC per iteracja per model

    for i in range(N_ITER):
        rs = BASE_SEED + i + 1  # +1 zeby nie pokryc sie z holdout seed
        res = run_iteration(ds, y_encoded, X_train_raw, y_train, i, random_state=rs)
        auc_history.append({'iter': i, **res['aucs']})

        feat = res['feature_names']
        combined = np.asarray(res['gene_iter_score'], dtype=float)

        # tylko top-K genow z tej iteracji (wg sum_model(SHAP*AUC))
        k = min(TOP_K_PER_ITER, len(feat))
        top_idx = np.argsort(combined)[::-1][:k]
        top_mask = np.zeros(len(feat), dtype=bool)
        top_mask[top_idx] = True

        for j, gene in enumerate(feat):
            if top_mask[j]:
                gene_scores_all[gene].append(float(combined[j]))

        for model_name, scores in res['per_model_scores'].items():
            auc_m = res['aucs'][model_name]
            scores = np.asarray(scores, dtype=float)
            for j, gene in enumerate(feat):
                if top_mask[j]:
                    gene_scores_per_model[model_name][gene].append(
                        float(scores[j]) * auc_m
                    )

    # ── rozklad empiryczny + CI per gen (laczny score) ──────────
    rng = np.random.default_rng(BASE_SEED)
    records = []
    for gene, vals in gene_scores_all.items():
        ci = compute_ci(vals, level=CI_LEVEL, n_boot=N_BOOTSTRAP, rng=rng)
        records.append({
            'gene': gene,
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
        .sort_values(['n_iter_seen', 'mean_score'], ascending=[False, False])
        .reset_index(drop=True)
    )
    stability_df.index += 1
    stability_df.index.name = 'rank'

    print(f"\nTotal unique genes seen: {len(stability_df)}")
    print("\nTop 30 genow (sort: n_iter_seen -> mean_score):")
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

    top_genes = [g for g in stability_df['gene'].head(TOP_K_FINAL).tolist()
                 if g in all_feat]
    print(f"Uzywam top-{TOP_K_FINAL} genow (obecnych w pipeline): {len(top_genes)}")

    X_tr_sel = X_tr_df[top_genes].values
    X_te_sel = X_te_df[top_genes].values
    spw = Counter(y_train)[0] / Counter(y_train)[1]

    final_aucs = {}
    for name, model in [
        ('LogReg', LogisticRegression(solver='saga', max_iter=15000,
                                      class_weight='balanced', fit_intercept=True)),
        ('XGB',    XGBClassifier(scale_pos_weight=spw, n_estimators=500,
                                 random_state=BASE_SEED, verbosity=0)),
        ('LGBM',   LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                  random_state=42, verbose=-1)),
    ]:
        model.fit(X_tr_sel, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_te_sel)[:, 1])
        final_aucs[name] = auc
        print(f"  {name:6s}  AUC = {auc:.4f}")
    print(f"  MEAN    AUC = {np.mean(list(final_aucs.values())):.4f}")

    pd.DataFrame([final_aucs]).to_csv('random_resampling_holdout_auc.csv', index=False)
    print("  -> random_resampling_holdout_auc.csv")

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
    axes[0].set_xlabel(f'sum_model(SHAP*AUC)  -- mean +/- {int(CI_LEVEL*100)}% empirical CI')
    axes[0].set_title(f'Top 30 genow (N_ITER={N_ITER})')

    # (b) rozklad empiryczny top-10 (boxplot)
    top10_genes = top30['gene'].head(10).tolist()
    data_box = [gene_scores_all[g] for g in top10_genes]
    axes[1].boxplot(data_box, vert=False, tick_labels=top10_genes, showmeans=True)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('sum_model(SHAP*AUC)  (rozklad empiryczny)')
    axes[1].set_title('Top 10 -- rozklad po iteracjach')

    plt.tight_layout()
    plt.savefig('gene_stability_random_resampling.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()
