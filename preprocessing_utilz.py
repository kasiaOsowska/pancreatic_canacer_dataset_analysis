from scipy.stats import mannwhitneyu
from utilz import *
from sklearn.base import BaseEstimator, TransformerMixin


class PValueReductor(BaseEstimator, TransformerMixin):
    def __init__(self, p_threshold=0.005):
        self.selected_genes_ = []
        self.p_threshold = p_threshold

    def fit(self, X, y):

        for gene in X.columns:
            X_gene = X[gene]
            X_healthy = X_gene[y == 0]
            X_cancer = X_gene[y == 1]

            u_stat, p_val = mannwhitneyu(X_healthy, X_cancer, alternative="two-sided")

            if p_val < self.p_threshold:
                self.selected_genes_.append(gene)

        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after PValueReductor: ", X.shape)
        return X

class VarianceExpressionReductor(BaseEstimator, TransformerMixin):
    def __init__(self, v_threshold=0.1):
        self.selected_genes_ = None
        self.v_threshold = v_threshold

    def fit(self, X, y=None):
        variance_per_gene = X.var(axis=0)
        self.selected_genes_ = variance_per_gene[variance_per_gene > self.v_threshold].index
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after VarianceExpressionReductor: ", X.shape)
        return X

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

class MinValueAdjustment(BaseEstimator, TransformerMixin):
    def __init__(self, method = "subtract"):
        self.method = method
        self.min_value = 0

    def fit(self, X, y=None):
        self.min_value = X.min().min()
        print("min value: ", self.min_value)
        return self

    def transform(self, X):
        match self.method:
            case "subtract":
                X[X == self.min_value] = X - self.min_value
            case "subtract_all":
                X = X - self.min_value
            case _:
                print(f"Unknown method: {self.method}, skipping")
        print("data shape after MinValueAdjustment: ", X.shape)
        return X
