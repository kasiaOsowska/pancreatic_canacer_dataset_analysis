from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Lasso
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, LabelEncoder


class AnovaReductor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_genes_ = None

    def fit(self, X, y=None):
        F, p = f_classif(X, y)
        self.selected_genes_ = X.columns[p < 0.05]
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after AnovaReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class MeanExpressionReductor(BaseEstimator, TransformerMixin):
    def __init__(self, mean_threshold=3):
        self.selected_genes_ = None
        self.mean_threshold = mean_threshold

    def fit(self, X, y=None):
        mean_per_gene = X.mean(axis=0)
        self.selected_genes_ = mean_per_gene[mean_per_gene > self.mean_threshold].index
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


class CovariatesBiasReductor(BaseEstimator, TransformerMixin):
    def __init__(self, covariate: pd.Series, p_thresh=1e-3):
        self.covariate = covariate
        self.p_thresh = p_thresh
        self.selected_genes_ = None

    def __sklearn_clone__(self):
        return copy.copy(self)

    def fit(self, X, y=None):
        import statsmodels.api as sm
        from statsmodels.stats.multitest import multipletests

        cov = self.covariate.loc[X.index].copy()
        valid_idx = cov.dropna().index
        cov = pd.Series(LabelEncoder().fit_transform(cov.loc[valid_idx]), index=valid_idx, dtype=float)

        covs = pd.DataFrame({'covariate': cov}, index=valid_idx)
        if y is not None:
            y_sub = pd.Series(LabelEncoder().fit_transform(pd.Series(y, index=X.index).loc[valid_idx]), index=valid_idx,
                              dtype=float)
            covs['disease'] = y_sub.values

        covs_matrix = sm.add_constant(covs)
        genes = list(X.columns)
        p_values = []

        for gene in genes:
            model = sm.OLS(X.loc[valid_idx, gene].values, covs_matrix.values).fit()
            p_values.append(model.pvalues[1])

        _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
        self.selected_genes_ = [g for g, p in zip(genes, p_adj) if p >= self.p_thresh]
        return self

    def transform(self, X):
        print("data shape after CovariatesBiasReductor: ", X.shape)
        return X[self.selected_genes_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)

class HighDispersionReductor(BaseEstimator, TransformerMixin):
    def __init__(self, quantile=0.95):
        self.quantile = quantile
        self.selected_genes_ = None

    def fit(self, X, y=None):
        dispersion = X.var() / (X.mean() + 1e-8)
        threshold = dispersion.quantile(self.quantile)
        self.selected_genes_ = dispersion[dispersion <= threshold].index
        return self

    def transform(self, X):
        print("data shape after HighDispersionReductor: ", X.shape)
        return X[self.selected_genes_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)