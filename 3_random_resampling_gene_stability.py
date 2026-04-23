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
INNER_TEST_SIZE     = 0.2     # rozmiar "wewnetrznego" val w kazdej iteracji
TOP_K_PER_ITER      = 1000     # ile top genow zapisywac z kazdej iteracji
TOP_K_FINAL         = 12      # ile top genow walidowac na holdout test
ANOVA_PERCENTILE    = 5
MEAN_PERCENTILE     = 5
WITHIN_GROUP_VAR_P  = 95
ANOVA_FDR_THRESHOLD = 0.1
CI_LEVEL            = 0.95    # przedzial ufnosci (np. 0.95 -> 2.5%/97.5%)
N_BOOTSTRAP         = 2000    # liczba bootstrap-resamplowan dla CI sredniej
BASE_SEED           = 2137
SCORING_METHOD      = 'anova_f'   # opcje: 'shap_auc', 'anova_f'


def build_pipeline(ds, y_encoded):
    sex_numeric = ds.sex.map({"F": 0, "M": 1})
    return Pipeline([
        ('ConstantExpressionReductor', ConstantExpressionReductor()),
        #('AnovaFDRReductor',           AnovaFdrReductor(alpha=ANOVA_FDR_THRESHOLD)),
        #('AnovaReductor',              AnovaReductor(percentile=ANOVA_PERCENTILE)),
        #('WithinGroupVarianceReductor',WithinGroupVarianceReductor(WITHIN_GROUP_VAR_P)),
        #('MeanExpressionReductor',     MeanExpressionReductor(percentile=MEAN_PERCENTILE)),
        ('SexBiasReductor',             CovariatesBiasReductor(covariate=sex_numeric)),
        ('AgeBiasReductor',            CovariatesResidualTransformer(covariate=ds.age, labels=y_encoded)),
        ('scaler',                     StandardScaler()),
    ])


def snr(shap_matrix):
    """|mean(SHAP)| / std(SHAP) per gene."""
    eps = 1e-10
    return np.abs(shap_matrix.mean(axis=0)) / (shap_matrix.std(axis=0) + eps)


def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


def score_shap_auc(X_tr, X_te, y_tr, y_te, feature_names):
    """
    Trenuje 3 modele (LogReg, XGB, LGBM), liczy SHAP na inner-val i zwraca:
      gene_iter_score = sum_model( minmax(SNR(|SHAP|)) * AUC_model )
    """
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

    # normalization
    n_lr = minmax_norm(snr(sv_lr))
    n_xgb = minmax_norm(snr(sv_xgb))
    n_lgbm = minmax_norm(snr(sv_lgbm))

    gene_iter_score = n_lr * auc_lr + n_xgb * auc_xgb + n_lgbm * auc_lgbm

    return {
        'gene_iter_score': gene_iter_score,
        'aucs': {'logreg': auc_lr, 'xgb': auc_xgb, 'lgbm': auc_lgbm},
        'per_model_scores': {
            'logreg': n_lr, 'xgb': n_xgb, 'lgbm': n_lgbm,
        },
    }


def score_anova_f(X_tr, X_te, y_tr, y_te, feature_names):
    """
    Liczy statystyke F z one-way ANOVA per gen (model-free).
    Nie wymaga trenowania zadnego klasyfikatora -> aucs/per_model_scores sa puste.
    """
    F, _ = f_classif(X_tr, y_tr)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        'gene_iter_score': F,
        'aucs': {},
        'per_model_scores': {},
    }


# rejestr metod scoringu -- dodaj nowa funkcje tutaj, zeby byla wybieralna
SCORERS = {
    'shap_auc': score_shap_auc,
    'anova_f':  score_anova_f,
}


def run_iteration(ds, y_encoded, X_train_raw, y_train, iteration_idx, random_state,
                  scoring_method=SCORING_METHOD):

    X_tr_raw, X_te_raw, y_tr, y_te = ds.get_train_test_valid_split(
        X_train_raw, y_train, test_size=INNER_TEST_SIZE, valid_size=0,
        random_state=random_state, return_valid=False,
    )

    pipe = build_pipeline(ds, y_encoded)
    X_tr = pipe.fit_transform(X_tr_raw, y_tr)
    X_te = pipe.transform(X_te_raw)
    feature_names = list(pipe.named_steps['SexBiasReductor'].selected_genes_)

    if scoring_method not in SCORERS:
        raise ValueError(f"Unknown scoring method: {scoring_method!r}. "
                         f"Available: {list(SCORERS)}")
    scorer = SCORERS[scoring_method]
    scored = scorer(X_tr, X_te, y_tr, y_te, feature_names)

    aucs = scored.get('aucs', {})
    auc_str = " ".join(f"{k}={v:.4f}" for k, v in aucs.items()) if aucs else "n/a"
    print(f"[iter {iteration_idx+1:>3}/{N_ITER}]  "
          f"inner_tr={len(X_tr_raw)}  inner_val={len(X_te_raw)}  "
          f"feat={len(feature_names)}  "
          f"method={scoring_method}  AUC {auc_str}")

    return {
        'feature_names': feature_names,
        'gene_iter_score': scored['gene_iter_score'],
        'aucs': aucs,
        'per_model_scores': scored.get('per_model_scores', {}),
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
        res = run_iteration(ds, y_encoded, X_train_raw, y_train, i, random_state=rs,
                            scoring_method=SCORING_METHOD)
        if res['aucs']:
            auc_history.append({'iter': i, **res['aucs']})

        feat = res['feature_names']
        combined = np.asarray(res['gene_iter_score'], dtype=float)

        # tylko top-K genow z tej iteracji (wg wybranej metody scoringu)
        k = min(TOP_K_PER_ITER, len(feat))
        top_idx = np.argsort(combined)[::-1][:k]
        top_mask = np.zeros(len(feat), dtype=bool)
        top_mask[top_idx] = True

        for j, gene in enumerate(feat):
            if top_mask[j]:
                gene_scores_all[gene].append(float(combined[j]))

        for model_name, scores in res['per_model_scores'].items():
            auc_m = res['aucs'].get(model_name, 1.0)
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
            'score': (len(vals)/N_ITER)*ci['mean'],
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

    # ── SelectKBest(f_classif): wybor TOP_K_FINAL genow ────────
    # UWAGA: fit WYLACZNIE na train; test NIE jest uzywany do selekcji.
    k = min(TOP_K_FINAL, X_tr_full.shape[1])
    skb = SelectKBest(f_classif, k=k).fit(X_tr_full, y_train)
    support_mask = skb.get_support()
    f_scores = np.nan_to_num(skb.scores_, nan=0.0, posinf=0.0, neginf=0.0)

    selected_pairs = sorted(
        [(g, float(f_scores[i])) for i, g in enumerate(all_feat) if support_mask[i]],
        key=lambda gs: gs[1], reverse=True,
    )
    top_genes = [g for g, _ in selected_pairs]
    print(f"Po SelectKBest(f_classif, k={k}) wybrano {len(top_genes)} genow "
          f"(ranking wg statystyki F):")
    for rank, (g, f) in enumerate(selected_pairs, 1):
        print(f"  rank={rank:>2}  F={f:8.3f}  {g}")

    X_tr_sel = X_tr_df[top_genes].values
    X_te_sel = X_te_df[top_genes].values
    spw = Counter(y_train)[0] / Counter(y_train)[1]

    final_aucs = {}
    model_weights = {}       # znormalizowana |waga/importance| per gen, per model
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

        # waga/waznosc cech z modelu (|coef_| dla liniowych, feature_importances_ dla drzew)
        if hasattr(model, 'coef_'):
            w = np.abs(np.ravel(model.coef_))
        else:
            w = np.asarray(model.feature_importances_, dtype=float)
        # normalizacja do sumy=1 zeby modele byly porownywalne
        total = w.sum()
        model_weights[name] = w / total if total > 0 else w

    print(f"  MEAN    AUC = {np.mean(list(final_aucs.values())):.4f}")

    # ── ocena przez srednia wage cech w modelach ────────────────
    mean_weight = np.mean(np.stack(list(model_weights.values()), axis=0), axis=0)
    weight_df = pd.DataFrame({
        'gene': top_genes,
        'mean_weight': mean_weight,
        **{f'weight_{n}': model_weights[n] for n in model_weights},
    }).sort_values('mean_weight', ascending=False).reset_index(drop=True)
    weight_df.index += 1
    weight_df.index.name = 'rank'
    weight_df.to_csv('gene_final_mean_weights.csv', index=True)

    print(f"\nRanking wybranych genow wg SREDNIEJ WAGI w modelach "
          f"(LogReg |coef|, XGB/LGBM feature_importances, znormalizowane do sumy=1):")
    print(weight_df.to_string())


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