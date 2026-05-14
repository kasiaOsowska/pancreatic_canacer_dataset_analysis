"""
3. Random Resampling Gene Stability Analysis (Elastic Net)

Schemat:
0. Holdout: jednorazowy podzial train/test
1. W kazdej iteracji: losowy subsplit w obrebie train + Elastic Net
2. Score per gen = freq_selekcji * mean(|coef|)
3. Top-12 genow walidowane na holdout test (incremental AUC)
"""

import warnings

from utilz.multi_residual_bootstrap import MultiCovariateResidualBootstrapTransformer, build_covariates

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
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
    ConstantExpressionReductor, AnovaFdrReductor,
    WithinGroupVarianceReductor, MeanExpressionReductor, AgeResidualBootstrapTransformer,
    SexResidualBootstrapTransformer, Log2FCReductor,
)

meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

N_ITER            = 10
HOLDOUT_TEST_SIZE = 0.2
VALID_SIZE = 0.2
INNER_TEST_SIZE   = 0.1
TOP_K_FINAL       = 20
ENET_C            = 0.1
ENET_L1_RATIO     = 0.9
BASE_SEED         = 2137
ANOVA_FDR_THRESHOLD = 0.1
LOG2FC_THRESHOLD  = np.log2(1.2)

def build_prepipeline(ds, labels=None):
    cov = build_covariates(ds.meta)
    return Pipeline([
        ('ConstantExpressionReductor', ConstantExpressionReductor()),
        ('multi_resid', MultiCovariateResidualBootstrapTransformer(
            covariates=cov, labels=labels,
            n_bootstrap=500, fdr_alpha=0.05, min_r2=0.05, cv_threshold_pct=30.0,
        )),
    ])


def build_iter_pipeline():
    return Pipeline([
        ('AnovaFDRReductor', AnovaFdrReductor(alpha=ANOVA_FDR_THRESHOLD)),
        ('Log2FCReductor', Log2FCReductor(min_abs_log2fc=LOG2FC_THRESHOLD)),
        ('WithinGroupVarianceReductor', WithinGroupVarianceReductor(alpha=0.05)),
        ('scaler', StandardScaler()),
    ])


def run_iteration(ds, X_train_pre, y_train, iteration_idx, random_state):
    X_tr_raw, _, y_tr, _ = ds.get_train_test_valid_split(
        X_train_pre, y_train, test_size=INNER_TEST_SIZE, valid_size=0,
        random_state=random_state, return_valid=False,
    )
    pipe = build_iter_pipeline()
    X_tr = pipe.fit_transform(X_tr_raw, y_tr)
    feature_names = list(pipe.named_steps['WithinGroupVarianceReductor'].selected_genes_)

    enet = LogisticRegression(
        penalty='elasticnet', solver='saga',
        l1_ratio=ENET_L1_RATIO, C=ENET_C,
        class_weight='balanced',
        max_iter=20000, random_state=random_state,
    ).fit(X_tr, y_tr)

    coefs = enet.coef_[0]
    n_selected = int((coefs > 0).sum())
    print(f"[iter {iteration_idx+1:>3}/{N_ITER}]  n_tr={len(X_tr_raw)}  "
          f"feat={len(feature_names)}  selected={n_selected}")

    return feature_names, coefs


def main():
    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    print("Class distribution:\n", y_encoded.value_counts().sort_index())

    X_train_raw, X_test_raw, y_train, y_test = ds.get_train_test_valid_split(
        ds.X, y_encoded, test_size=HOLDOUT_TEST_SIZE, valid_size=VALID_SIZE,
        random_state=BASE_SEED, return_valid=False,
    )
    print(f"\nTrain: {len(X_train_raw)}  Test: {len(X_test_raw)}")
    print("\n[pre-bootstrap] fit ConstantExpressionReductor + Age/Sex debias")

    prepipe = build_prepipeline(ds, labels=y_train)
    X_train_pre = prepipe.fit_transform(X_train_raw, y_train)
    X_test_pre  = prepipe.transform(X_test_raw)
    print(f"[pre-bootstrap] preprocessing: train={X_train_pre.shape}  test={X_test_pre.shape}\n")

    gene_coefs = defaultdict(list)
    for i in range(N_ITER):
        feat, coefs = run_iteration(ds, X_train_pre, y_train, i,
                                    random_state=BASE_SEED + i + 1)
        for j, gene in enumerate(feat):
            if coefs[j] > 0:
                gene_coefs[gene].append(coefs[j])

    records = [
        {
            'gene': gene,
            'n_iter_seen': len(vals),
            'freq': len(vals) / N_ITER,
            'mean_abs_coef': float(np.mean(vals)),
            'score': (len(vals) / N_ITER) * float(np.mean(vals)),
        }
        for gene, vals in gene_coefs.items()
    ]
    stability_df = (
        pd.DataFrame(records)
        .sort_values('score', ascending=False)
        .reset_index(drop=True)
    )
    stability_df.index += 1
    stability_df.index.name = 'rank'
    stability_df.to_csv('gene_stability_summary.csv', index=True)

    print(f"\nTotal unique selected genes: {len(stability_df)}")
    print(f"\nTop {TOP_K_FINAL} genow (score = freq * mean|coef|):")
    print(stability_df.head(TOP_K_FINAL).to_string())

    print(f"\n{'='*60}\nholdout test (n={len(X_test_raw)})\n{'='*60}")

    pipe_full = build_iter_pipeline()
    X_tr_full = pipe_full.fit_transform(X_train_pre, y_train)
    X_te_full = pipe_full.transform(X_test_pre)
    all_feat = list(pipe_full.named_steps['MeanExpressionReductor'].selected_genes_)

    X_tr_df = pd.DataFrame(X_tr_full, columns=all_feat)
    X_te_df = pd.DataFrame(X_te_full, columns=all_feat)

    ranked_genes = [g for g in stability_df['gene'].head(TOP_K_FINAL).tolist()
                    if g in X_tr_df.columns]
    spw = Counter(y_train)[0] / Counter(y_train)[1]

    def make_models():
        return [
            ('LogReg', LogisticRegression(solver='saga', max_iter=15000,
                                          class_weight='balanced')),
            ('XGB',    XGBClassifier(scale_pos_weight=spw, n_estimators=500,
                                     random_state=BASE_SEED, verbosity=0)),
            ('LGBM',   LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                      random_state=42, verbose=-1)),
        ]

    rows = []
    for k in range(1, len(ranked_genes) + 1):
        genes_k = ranked_genes[:k]
        X_tr_sel = X_tr_df[genes_k].values
        X_te_sel = X_te_df[genes_k].values
        row = {'k': k, 'gene_added': genes_k[-1]}
        aucs_k = []
        for name, model in make_models():
            model.fit(X_tr_sel, y_train)
            auc = roc_auc_score(y_test, model.predict_proba(X_te_sel)[:, 1])
            row[name] = auc
            aucs_k.append(auc)
        row['MEAN'] = float(np.mean(aucs_k))
        rows.append(row)
        print(f"  k={k:>2}  + {row['gene_added']:20s}  "
              f"LogReg={row['LogReg']:.4f}  XGB={row['XGB']:.4f}  "
              f"LGBM={row['LGBM']:.4f}  MEAN={row['MEAN']:.4f}")

    incremental_df = pd.DataFrame(rows)
    incremental_df.to_csv('gene_incremental_aucs.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ks = incremental_df['k'].values
    for col, color in [('LogReg', '#6366f1'), ('XGB', '#10b981'),
                       ('LGBM', '#f59e0b'), ('MEAN', '#1f2937')]:
        ax.plot(ks, incremental_df[col].values, marker='o', label=col, color=color,
                lw=2.5 if col == 'MEAN' else 1.5,
                ls='--' if col == 'MEAN' else '-')
    ax.set_xlabel('Liczba genow (top-k wg ENet stability)')
    ax.set_ylabel('AUC na holdout test')
    ax.set_title('AUC vs liczba genow')
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('gene_incremental_aucs.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()