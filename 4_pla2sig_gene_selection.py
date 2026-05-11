"""
Replikacja metody selekcji genow z PLA2Sig (Ji et al., BJC 2025).
DEG (log2FC + Mann-Whitney) -> LASSO 100-fold CV (lambda.1se)
-> inkrementalne top-k -> binomial GLM.
"""

import warnings;

from utilz.multi_residual_bootstrap import MultiCovariateResidualBootstrapTransformer, build_covariates

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY
from utilz.preprocessing_utilz import (
    ConstantExpressionReductor, Log2FCReductor, MannWhitneyReductor,
)

meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

HOLDOUT_TEST_SIZE = 0.4
BASE_SEED         = 2137
LOG2FC_THRESHOLD  = np.log2(1.2)
DEG_PVAL          = 0.05
LASSO_CV_FOLDS    = 100
LASSO_N_LAMBDAS   = 50
TOP_K_MAX         = 20
INCR_CV_FOLDS     = 10
DELTA_AUC_SIG     = 0.005


def lasso_cv_lambda_1se(X, y, n_folds=LASSO_CV_FOLDS,
                        n_lambdas=LASSO_N_LAMBDAS, seed=BASE_SEED):
    n_folds = min(n_folds, np.bincount(y.astype(int)).min())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv = LogisticRegressionCV(
        Cs=n_lambdas, cv=skf, penalty='l1', solver='saga',
        scoring='neg_log_loss', class_weight='balanced',
        max_iter=20000, n_jobs=-1, random_state=seed, refit=False,
    ).fit(X, y)

    losses = -cv.scores_[1]
    mean_loss = losses.mean(axis=0)
    se_loss = losses.std(axis=0, ddof=1) / np.sqrt(n_folds)
    k_min = int(np.argmin(mean_loss))
    threshold = mean_loss[k_min] + se_loss[k_min]
    k_1se = int(np.where(mean_loss <= threshold)[0].min())
    C_1se = float(cv.Cs_[k_1se])
    print(f"[LASSO] {n_folds}-fold, lambda.1se -> C={C_1se:.5g} "
          f"loss={mean_loss[k_1se]:.4f} (lambda.min loss={mean_loss[k_min]:.4f}+-{se_loss[k_min]:.4f})")

    final = LogisticRegression(
        penalty='l1', solver='saga', C=C_1se, max_iter=20000,
        class_weight='balanced', random_state=seed,
    ).fit(X, y)
    return final.coef_.ravel()


def incremental_topk_eval(X, y, ranked_idx, k_max=TOP_K_MAX,
                          n_folds=INCR_CV_FOLDS, seed=BASE_SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for k in range(1, min(k_max, len(ranked_idx)) + 1):
        idx = ranked_idx[:k]
        aucs = []
        for tr, va in skf.split(X, y):
            mdl = LogisticRegression(
                max_iter=20000, class_weight='balanced', random_state=seed,
            ).fit(X[np.ix_(tr, idx)], y[tr])
            aucs.append(roc_auc_score(y[va], mdl.predict_proba(X[np.ix_(va, idx)])[:, 1]))
        rows.append({'k': k, 'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs, ddof=1)})
        print(f"  top-{k:>2}  CV AUC = {rows[-1]['auc_mean']:.4f}+-{rows[-1]['auc_std']:.4f}")
    return pd.DataFrame(rows)


def pick_minimal_k(incr_df, delta=DELTA_AUC_SIG):
    auc = incr_df['auc_mean'].values
    return int(incr_df['k'].values[auc >= auc.max() - delta].min())


def main():
    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})
    y_enc = pd.Series(LabelEncoder().fit_transform(ds.y), index=ds.y.index)

    X_tr_raw, X_te_raw, y_train, y_test = ds.get_train_test_valid_split(
        ds.X, y_enc, test_size=HOLDOUT_TEST_SIZE, valid_size=0,
        random_state=BASE_SEED, return_valid=False,
    )
    print(f"Train: {len(X_tr_raw)}  cancer={int(y_train.sum())} ctrl={int((y_train==0).sum())}")
    print(f"Test:  {len(X_te_raw)}  cancer={int(y_test.sum())}  ctrl={int((y_test==0).sum())}")

    print("\n=== KROK 1: DEG ===")
    """
            ('multi_resid', MultiCovariateResidualBootstrapTransformer(
        covariates=cov, labels=y_train,
        n_bootstrap=500, fdr_alpha=0.1, min_r2=0.05, cv_threshold_pct=30.0,
        )),
    """
    cov = build_covariates(ds.meta)
    deg_pipe = Pipeline([
        ('const',  ConstantExpressionReductor()),
        ('log2fc', Log2FCReductor(min_abs_log2fc=LOG2FC_THRESHOLD)),
        ('pval',   MannWhitneyReductor(alpha=DEG_PVAL)),
    ])
    X_tr_deg_df = deg_pipe.fit_transform(X_tr_raw, y_train)
    X_te_deg_df = deg_pipe.transform(X_te_raw)
    deg_genes = list(X_tr_deg_df.columns)

    print("\n=== KROK 2: LASSO ===")
    scaler = StandardScaler().fit(X_tr_deg_df.values)
    X_tr_z = scaler.transform(X_tr_deg_df.values)
    X_te_z = scaler.transform(X_te_deg_df.values)
    coefs = lasso_cv_lambda_1se(X_tr_z, y_train.values)
    print(f"[LASSO] niezerowe wsp.: {(coefs != 0).sum()} / {len(coefs)}")

    rank_df = (pd.DataFrame({'gene': deg_genes, 'coef': coefs})
               .assign(abs_coef=lambda d: d['coef'].abs())
               .query('coef != 0')
               .sort_values('abs_coef', ascending=False)
               .reset_index(drop=True))
    print("\nTop 20 wg |coef|:")
    print(rank_df.head(20)[['gene', 'coef']].to_string())

    print("\n=== KROK 3: top-k ===")
    g2c = {g: i for i, g in enumerate(deg_genes)}
    ranked_idx = [g2c[g] for g in rank_df['gene'].tolist()]
    incr_df = incremental_topk_eval(X_tr_z, y_train.values, ranked_idx)
    k_star = pick_minimal_k(incr_df)
    minimal_genes = rank_df['gene'].head(k_star).tolist()
    print(f"\nk* = {k_star}; geny: {minimal_genes}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(incr_df['k'], incr_df['auc_mean'], yerr=incr_df['auc_std'],
                marker='o', capsize=3, label=f'CV AUC ({INCR_CV_FOLDS}-fold)')
    ax.axvline(k_star, color='red', ls='--', alpha=0.6, label=f'k* = {k_star}')
    ax.set(xlabel='Top-k genes', ylabel='AUC',
           title='PLA2Sig replication: top-k AUC on train CV')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    print("\n=== KROK 4: GLM ===")
    idx_min = [g2c[g] for g in minimal_genes]
    glm = LogisticRegression(
        penalty=None, solver='lbfgs', max_iter=20000,
        class_weight='balanced', random_state=BASE_SEED,
    ).fit(X_tr_z[:, idx_min], y_train.values)

    auc_tr = roc_auc_score(y_train.values, glm.predict_proba(X_tr_z[:, idx_min])[:, 1])
    auc_te = roc_auc_score(y_test.values,  glm.predict_proba(X_te_z[:, idx_min])[:, 1])
    print(f"coef:       {dict(zip(minimal_genes, glm.coef_.ravel().round(4)))}")
    print(f"intercept:  {glm.intercept_[0]:.4f}")
    print(f"Train AUC:  {auc_tr:.4f}")
    print(f"Holdout AUC:{auc_te:.4f}")


if __name__ == '__main__':
    main()
