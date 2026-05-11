import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

def find_stable_covariate_genes(
    X: pd.DataFrame,
    covariate: pd.Series,
    fdr_alpha: float = 0.05,
    n_bootstrap: int = 1000,
    cv_threshold_pct: float = 30.0,
    ci_level: float = 0.95,
    random_state: int = 2137,
    verbose: bool = True,):

    cov = pd.Series(covariate).reindex(X.index).astype(float)
    valid = cov.dropna().index
    X = X.loc[valid]
    cov_vals = cov.loc[valid].values
    n = len(cov_vals)

    if n < 5:
        return pd.DataFrame(columns=['gene', 'ci_lower', 'ci_upper', 'cv_pct'])

    expr_full = X.values  # (n, n_genes_all)
    gene_names_all = list(X.columns)

    # --- 1. FDR-BH pre-filter (per-gene p from OLS) ---
    pvals = np.empty(len(gene_names_all))
    for j in range(len(gene_names_all)):
        pvals[j] = stats.linregress(cov_vals, expr_full[:, j]).pvalue
    pvals = np.nan_to_num(pvals, nan=1.0)
    rejected, _, _, _ = multipletests(pvals, alpha=fdr_alpha, method='fdr_bh')
    cand_idx = np.where(rejected)[0]
    candidate_genes = [gene_names_all[i] for i in cand_idx]
    print(f"[stable-genes] after FDR-BH (alpha={fdr_alpha}): "
          f"{len(candidate_genes)}/{len(gene_names_all)}")

    if len(candidate_genes) == 0:
        return pd.DataFrame(columns=['gene', 'ci_lower', 'ci_upper', 'cv_pct'])

    expr_mat = expr_full[:, cand_idx]

    # --- 2. bootstrap  ---
    rng = np.random.default_rng(random_state)
    coefs_boot = np.empty((n_bootstrap, expr_mat.shape[1]))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        x = cov_vals[idx]
        y = expr_mat[idx, :]
        x_c = x - x.mean()
        y_c = y - y.mean(axis=0)
        denom = (x_c ** 2).sum()
        coefs_boot[b, :] = (x_c @ y_c) / denom if denom > 0 else 0.0

    # --- 3. selection: CI not intersecting 0 AND CV < threshold ---
    alpha = 1.0 - ci_level
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)

    median_coef = np.median(coefs_boot, axis=0)
    ci_low      = np.percentile(coefs_boot, lo_p, axis=0)
    ci_high     = np.percentile(coefs_boot, hi_p, axis=0)
    std_coef    = np.std(coefs_boot, axis=0)
    cv_pct      = np.where(np.abs(median_coef) > 1e-12,
                           std_coef / np.abs(median_coef) * 100,
                           np.inf)

    ci_excludes_zero = (ci_low > 0) | (ci_high < 0)
    cv_ok            = cv_pct < cv_threshold_pct
    stable_mask      = ci_excludes_zero & cv_ok

    stable_df = pd.DataFrame({
        'gene':     [candidate_genes[i] for i in np.where(stable_mask)[0]],
        'ci_lower': ci_low[stable_mask],
        'ci_upper': ci_high[stable_mask],
        'cv_pct':   cv_pct[stable_mask],
    }).reset_index(drop=True)

    if verbose:
        print(f"[stable-genes] stabilnych po bootstrap (CI!=0, CV<{cv_threshold_pct}%): "
              f"{len(stable_df)}")

    return stable_df

class _ResidualBootstrapBase(BaseEstimator, TransformerMixin):
    def __init__(self, covariate, labels=None,
                 fdr_alpha=0.05, n_bootstrap=1000,
                 cv_threshold_pct=30.0, ci_level=0.95,
                 random_state=2137, verbose=True):
        self.covariate = covariate
        self.labels = labels
        self.fdr_alpha = fdr_alpha
        self.n_bootstrap = n_bootstrap
        self.cv_threshold_pct = cv_threshold_pct
        self.ci_level = ci_level
        self.random_state = random_state
        # fitted attrs
        self.selected_genes_ = None
        self.stable_info_    = None
        self.coef_           = None
        self.intercept_      = None

    def _prepare_covariate(self, index):
        return pd.Series(self.covariate).reindex(index).astype(float)

    def fit(self, X: pd.DataFrame, y=None):
        cov = self._prepare_covariate(X.index)
        if self.labels is not None:
            lab = pd.Series(self.labels).reindex(X.index)
            base_idx = lab[lab == 0].index
        else:
            base_idx = X.index

        valid = cov.loc[base_idx].dropna().index
        if len(valid) < 5:
            self.selected_genes_ = []
            self.stable_info_ = pd.DataFrame(columns=['gene','ci_lower','ci_upper','cv_pct'])
            self.coef_ = np.zeros(0)
            self.intercept_ = np.zeros(0)
            return self

        # 1. bootstrap stable genes
        stable_df = find_stable_covariate_genes(
            X.loc[valid], cov.loc[valid],
            fdr_alpha=self.fdr_alpha,
            n_bootstrap=self.n_bootstrap,
            cv_threshold_pct=self.cv_threshold_pct,
            ci_level=self.ci_level,
            random_state=self.random_state
        )
        self.stable_info_ = stable_df
        self.selected_genes_ = stable_df['gene'].tolist()

        if len(self.selected_genes_) == 0:
            self.coef_ = np.zeros(0)
            self.intercept_ = np.zeros(0)
            return self

        # OLS per gen przez sklearn.LinearRegression (multi-output):
        # x = age (n, 1), Y = ekspresja (n, n_genes); jedno wywolanie fitu
        # robi niezalezne OLS dla kazdej kolumny Y.
        x = cov.loc[valid].values.reshape(-1, 1)               # (n, 1)
        Y = X.loc[valid, self.selected_genes_].values          # (n, n_genes)
        lr = LinearRegression().fit(x, Y)
        self.coef_      = lr.coef_.ravel()                     # (n_genes,)
        self.intercept_ = lr.intercept_                        # (n_genes,)
        print(f"  [{type(self).__name__}] fit OLS on {len(valid)} probs "
              f"for {len(self.selected_genes_)} stable genes "
              f"(bootstrap N={self.n_bootstrap})")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(self.selected_genes_) == 0:
            return X.copy()
        cov = self._prepare_covariate(X.index).values
        cov_filled = np.where(np.isnan(cov), 0.0, cov)
        predicted = np.outer(cov_filled, self.coef_) + self.intercept_
        predicted[np.isnan(cov)] = 0.0
        present = [g for g in self.selected_genes_ if g in X.columns]
        if len(present) == 0:
            return X.copy()
        idx_in_full = [self.selected_genes_.index(g) for g in present]
        predicted_present = predicted[:, idx_in_full]

        result = X.copy()
        result[present] = X[present].values - predicted_present
        print(f"data shape after {type(self).__name__}: {result.shape}")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [],
                          dtype=object)