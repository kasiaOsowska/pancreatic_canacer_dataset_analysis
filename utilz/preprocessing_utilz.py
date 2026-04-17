import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import f_classif


class AnovaFdrReductor(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.selected_genes_ = None

    def fit(self, X, y=None):
        F, p = f_classif(X, y)
        # Benjamini-Hochberg FDR correction
        _, p_corrected, _, _ = multipletests(p, alpha=self.alpha, method='fdr_bh')
        self.selected_genes_ = X.columns[p_corrected < self.alpha]
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after AnovaFdrReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)



class MeanExpressionReductor(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=25):
        self.selected_genes_ = None
        self.percentile = percentile

    def fit(self, X, y=None):
        mean_per_gene = X.mean(axis=0)
        threshold = np.percentile(mean_per_gene, self.percentile)
        self.selected_genes_ = mean_per_gene[mean_per_gene > threshold].index
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after MeanExpressionReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class ConstantExpressionReductor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_genes_ = None

    def fit(self, X, y=None):
        num_unique = X.nunique(dropna=True)
        self.selected_genes_ = num_unique[num_unique > 1].index
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after ConstantExpressionReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class AnovaReductor(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=95):
        self.selected_genes_ = None
        self.percentile = percentile

    def fit(self, X, y=None):
        f_scores, _ = f_classif(X, y)
        threshold = np.percentile(f_scores, self.percentile)
        self.selected_genes_ = X.columns[f_scores>threshold]
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after AnovaReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class Log2FCReductor(BaseEstimator, TransformerMixin):
    def __init__(self, min_abs_log2fc=1.0):
        self.selected_genes_ = None
        self.min_abs_log2fc = min_abs_log2fc

    def fit(self, X, y=None):
        group_means = X.groupby(y).mean()
        class0, class1 = group_means.index
        mean0 = group_means.loc[class0]
        mean1 = group_means.loc[class1]
        log2fc = np.log2((mean1 + 1e-6) / (mean0 + 1e-6))
        abs_log2fc = np.abs(log2fc)
        print(abs_log2fc.max())
        valid_genes = abs_log2fc >= self.min_abs_log2fc
        self.selected_genes_ = X.columns[valid_genes]
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after Log2FCReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class WithinGroupVarianceReductor(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=20):
        self.percentile = percentile
        self.selected_genes_ = None

    def fit(self, X: pd.DataFrame, y=None):

        y_s = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y.reindex(X.index)
        classes = y_s.unique()

        # pooled within-group variance per gene
        N = len(X)
        K = len(classes)
        wgv = pd.Series(0.0, index=X.columns)

        for cls in classes:
            mask = y_s == cls
            n_k = mask.sum()
            if n_k > 1:
                wgv += (n_k - 1) * X.loc[mask].var(ddof=1)

        wgv /= (N - K)

        threshold = np.percentile(wgv, self.percentile)
        self.selected_genes_ = wgv[wgv < threshold].index
        return self

    def transform(self, X: pd.DataFrame):
        X = X[self.selected_genes_]
        print("data shape after WithinGroupVarianceReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class CovariatesBiasReductor(BaseEstimator, TransformerMixin):

    def __init__(self, covariate: pd.Series, p_thresh: float = 0.05, beta_thresh: float = None):
        self.covariate = covariate
        self.p_thresh = p_thresh
        self.beta_thresh = beta_thresh
        self.selected_genes_ = None

    def fit(self, X: pd.DataFrame, y=None):
        genes = list(X.columns)
        cov = pd.Series(self.covariate).reindex(X.index).astype(float)
        valid_idx = cov.dropna().index
        cov = cov.loc[valid_idx]
        covs = pd.DataFrame(index=valid_idx)
        covs["covariate"] = cov
        if y is not None:
            y_series = pd.Series(y).reindex(X.index)
            y_series = y_series.loc[valid_idx].astype(float)
            covs["disease"] = y_series

        covs_matrix = sm.add_constant(covs)
        cov_col_idx = list(covs_matrix.columns).index("covariate")
        p_values = []
        beta_values = []

        for gene in genes:
            expr = X.loc[valid_idx, gene].astype(float).values
            try:
                model = sm.OLS(expr, covs_matrix.values).fit()
                p = model.pvalues[cov_col_idx]
                beta = model.params[cov_col_idx]
            except Exception:
                p, beta = np.nan, np.nan
            p_values.append(p)
            beta_values.append(beta)

        p_values = np.array(p_values)
        beta_values = np.array(beta_values)

        p_values = np.nan_to_num(p_values, nan=1.0)
        _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
        keep = (p_adj >= self.p_thresh)
        if self.beta_thresh is not None:
            keep = keep & (np.abs(beta_values) <= self.beta_thresh)

        self.selected_genes_ = [g for g, k in zip(genes, keep) if k]
        return self

    def transform(self, X: pd.DataFrame):
        X = X[self.selected_genes_]
        print("data shape after CovariatesBiasReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class CovariatesResidualTransformer(BaseEstimator, TransformerMixin):
    """
    Dla kazdego genu dopasowuje regresje: expression ~ covariate
    tylko na zdrowych probkach i zastepuje ekspresje residuami.

    Parametry
    ---------
    covariate : pd.Series
        Wartosci kowarianty (np. wiek) indeksowane sample-ID.
    labels : pd.Series, optional
        Etykiety choroby (0=zdrowy, 1=chory) indeksowane sample-ID.
        Jesli podane, regresja jest uczona TYLKO na zdrowych (labels==0).
        Jesli None, regresja jest uczona na wszystkich probkach.
    """

    def __init__(self, covariate: pd.Series, labels: pd.Series = None):
        self.covariate = covariate
        self.labels = labels
        self.coef_ = None
        self.intercept_ = None
        self.selected_genes_ = None

    def _get_covariate(self, index):
        cov = self.covariate
        if not isinstance(cov, pd.Series):
            cov = pd.Series(cov)

        matched = cov.reindex(index)
        if matched.notna().sum() > 0:
            return matched.astype(float)

        print("  [CovariatesResidualTransformer] WARNING: index mismatch, using positional matching")
        return pd.Series(cov.values[:len(index)], index=index, dtype=float)

    def fit(self, X: pd.DataFrame, y=None):
        self.selected_genes_ = list(X.columns)
        cov = self._get_covariate(X.index)

        # etykiety choroby z __init__, NIE z pipeline'owego y
        if self.labels is not None:
            lab = self.labels.reindex(X.index)
            healthy_idx = lab[lab == 0].index
        else:
            healthy_idx = X.index

        cov_healthy = cov.loc[healthy_idx].dropna()
        valid_idx = cov_healthy.index

        cov_vals = cov_healthy.values
        X_valid  = X.loc[valid_idx].values

        X_design = np.column_stack([np.ones(len(cov_vals)), cov_vals])
        coeffs, _, _, _ = np.linalg.lstsq(X_design, X_valid, rcond=None)

        self.intercept_ = coeffs[0]
        self.coef_      = coeffs[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cov = self._get_covariate(X.index)
        cov_vals = cov.values

        cov_filled = np.where(np.isnan(cov_vals), 0.0, cov_vals)
        predicted  = np.outer(cov_filled, self.coef_) + self.intercept_

        missing_mask = np.isnan(cov_vals)
        predicted[missing_mask] = 0.0

        residuals = X[self.selected_genes_].values - predicted
        result = pd.DataFrame(residuals, index=X.index, columns=self.selected_genes_)
        print(f"data shape after CovariatesResidualTransformer: {result.shape}")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)