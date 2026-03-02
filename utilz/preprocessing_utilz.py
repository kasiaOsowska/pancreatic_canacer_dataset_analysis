from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Lasso
import pandas as pd
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
    def __init__(self, age, alpha=0.11, max_iter=15000):
        # pełny wektor wieku z ds.age
        self.age = pd.Series(age)
        self.alpha = alpha
        self.max_iter = max_iter
        self.selected_genes_ = None

    def fit(self, X, y=None):
        # oczekujemy DataFrame (krok przed scalerem, jak masz w pipeline)
        if not hasattr(X, "index"):
            raise TypeError("AgeBiasReductor expects X as pandas DataFrame.")

        # dopasuj age do próbek w X
        age = self.age.loc[X.index]

        # rzutuj na liczby, śmieci -> NaN
        age = pd.to_numeric(age, errors="coerce")

        # wyrzuć próbki z brakującym wiekiem
        mask = age.notna()
        X_sub = X.loc[mask]
        age_sub = age.loc[mask].astype(float).values

        # mało próbek? nie usuwaj nic
        if X_sub.shape[0] < 5:
            # w takiej sytuacji lepiej nic nie robić
            self.selected_genes_ = list(X.columns)
            return self

        model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        model.fit(X_sub.values, age_sub)

        coef = model.coef_.ravel()
        # geny NIEzwiązane z wiekiem (współczynnik 0)
        mask_not_age_related = np.abs(coef) == 0
        self.selected_genes_ = X.columns[mask_not_age_related].tolist()
        return self

    def transform(self, X):
        X_out = X[self.selected_genes_]
        print("data shape after AgeBiasReductor: ", X_out.shape)
        return X_out

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


class SexBiasReductor(BaseEstimator, TransformerMixin):
    def __init__(self, sex, p_thresh=1e-3, d_thresh=0.8):
        self.sex = pd.Series(sex)
        self.p_thresh = p_thresh
        self.d_thresh = d_thresh
        self.selected_genes_ = None

    def fit(self, X, y=None):
        sex = self.sex.loc[X.index]
        biased_genes = set()
        for gene in X.columns:
            X_gene = X[gene]
            X_female = X_gene[sex == "F"]
            X_male   = X_gene[sex == "M"]
            if len(X_female) == 0 or len(X_male) == 0:
                continue
            u_stat, p_val = mannwhitneyu(
                X_female, X_male, alternative="two-sided"
            )

            if p_val < self.p_thresh:
                # cohen's d
                mean_female = X_female.mean()
                mean_male = X_male.mean()
                n_f, n_m = len(X_female), len(X_male)
                std_f = X_female.std(ddof=1)
                std_m = X_male.std(ddof=1)
                pooled_std = np.sqrt(
                    ((n_f - 1) * std_f**2 + (n_m - 1) * std_m**2)
                    / (n_f + n_m - 2)
                )
                if pooled_std == 0:
                    continue
                cohen_d = (mean_female - mean_male) / pooled_std
                if abs(cohen_d) > self.d_thresh:
                    biased_genes.add(gene)

        self.selected_genes_ = [
            g for g in X.columns if g not in biased_genes
        ]

        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after SexBiasReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)
