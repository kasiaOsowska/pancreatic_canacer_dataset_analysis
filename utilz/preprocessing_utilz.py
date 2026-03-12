from sklearn.feature_selection import f_classif
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.feature_selection import SelectFdr, f_classif

class AnovaFdrReductor(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05):
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


class HighVarianceReductor(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=95):
        self.selected_genes_ = None
        self.percentile = percentile

    def fit(self, X, y=None):
        var_per_gene = X.var(axis=0)
        threshold = np.percentile(var_per_gene, self.percentile)
        self.selected_genes_ = var_per_gene[var_per_gene <= threshold].index
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after HighVarianceReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class CovariatesBiasReductor(BaseEstimator, TransformerMixin):

    def __init__(self, covariate: pd.Series, p_thresh: float = 0.05):
        self.covariate = covariate
        self.p_thresh = p_thresh
        self.selected_genes_ = None

    def fit(self, X: pd.DataFrame, y=None):

        genes = list(X.columns)
        cov = pd.Series(self.covariate).reindex(X.index).astype(float)
        missing_cov = cov.isna().sum()
        if missing_cov > 0:
            print(f"[WARN] {missing_cov} missing coves, skipped in fit()")

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
        for gene in genes:
            expr = X.loc[valid_idx, gene].astype(float).values
            try:
                model = sm.OLS(expr, covs_matrix.values).fit()
                p = model.pvalues[cov_col_idx]
            except Exception:
                p = np.nan
            p_values.append(p)
        p_values = np.array(p_values)
        p_values = np.nan_to_num(p_values, nan=1.0)
        _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
        self.selected_genes_ = [
            g for g, p in zip(genes, p_adj) if p >= self.p_thresh
        ]
        return self

    def transform(self, X: pd.DataFrame):
        print("data shape after CovariatesBiasReductor: ", X.shape)
        return X[self.selected_genes_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class MRMRReductor(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=100):
        self.selected_genes_ = None
        self.n_features = n_features

    def fit(self, X, y=None):
        from sklearn.feature_selection import f_classif

        genes = X.columns
        f_scores, _ = f_classif(X, y)
        relevance = pd.Series(f_scores, index=genes).fillna(0)
        corr_matrix = X.corr().abs()

        selected = []
        remaining = list(genes)
        first = relevance[remaining].idxmax()
        selected.append(first)
        remaining.remove(first)

        for _ in range(1, min(self.n_features, len(genes))):
            redundancy = corr_matrix.loc[remaining, selected].mean(axis=1)
            scores = relevance[remaining] - redundancy
            best_gene = scores.idxmax()
            selected.append(best_gene)
            remaining.remove(best_gene)

        self.selected_genes_ = selected
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after MRMRReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)