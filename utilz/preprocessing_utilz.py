from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


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

class AgeBiasReductor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_genes_ = None

    def fit(self, X, y=None):

        model = Lasso(alpha = 0.11, max_iter=10000)
        model.fit(X, y)
        coef = model.coef_.ravel()
        mask = np.abs(coef) == 0
        idx = np.where(mask)[0]
        self.selected_genes_ = idx
        return self

    def transform(self, X):
        X = X[:, self.selected_genes_]
        print("data shape after AgeBiasReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class NoneInformativeGeneReductor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_genes_ = None

    def fit(self, X, y=None):
        num_unique = X.nunique(dropna=True)
        self.selected_genes_ = num_unique[num_unique > 1].index
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after NoneInformativeGeneReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)
