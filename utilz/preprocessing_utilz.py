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
    Dla kazdego genu osobno dopasowuje regresje liniowa:
        expression ~ covariate
    i zastepuje wartosci ekspresji residuami:
        residual = observed_expression - predicted_expression

    Nie usuwa zadnych genow — zachowuje pelny zestaw cech.
    Parametry z fitu (slope, intercept) sa zapamietane i uzywane przy transform,
    dzieki czemu zbior testowy jest korygowany modelami wytrenowanymi na trainie.
    """

    def __init__(self, covariate: pd.Series):
        self.covariate = covariate
        self.coef_ = None       # slope per gene,     shape (n_genes,)
        self.intercept_ = None  # intercept per gene, shape (n_genes,)
        self.genes_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.genes_ = list(X.columns)
        cov = pd.Series(self.covariate).reindex(X.index).astype(float)

        # probki z brakujaca wartoscia kowariaty sa pomijane przy ficie
        valid_idx = cov.dropna().index
        cov_vals  = cov.loc[valid_idx].values  # (n,)
        X_valid   = X.loc[valid_idx].values    # (n, p)

        # X_design = [1, covariate] -> OLS dla wszystkich genow naraz
        X_design = np.column_stack([np.ones(len(cov_vals)), cov_vals])  # (n, 2)
        coeffs, _, _, _ = np.linalg.lstsq(X_design, X_valid, rcond=None)  # (2, p)

        self.intercept_ = coeffs[0]  # (p,)
        self.coef_      = coeffs[1]  # (p,)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cov = pd.Series(self.covariate).reindex(X.index).astype(float)
        cov_vals = cov.values  # (n,)

        # predicted = intercept + covariate * slope dla kazdej probki i genu
        cov_filled = np.where(np.isnan(cov_vals), 0.0, cov_vals)
        predicted  = np.outer(cov_filled, self.coef_) + self.intercept_  # (n, p)

        # tam gdzie kowariata brakuje — nie odejmujemy predykcji
        missing_mask = np.isnan(cov_vals)
        predicted[missing_mask] = 0.0

        residuals = X[self.genes_].values - predicted
        result = pd.DataFrame(residuals, index=X.index, columns=self.genes_)
        print("data shape after CovariatesResidualTransformer: ", result.shape)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.genes_, dtype=object)