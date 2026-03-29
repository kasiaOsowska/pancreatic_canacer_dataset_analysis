import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import f_classif

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