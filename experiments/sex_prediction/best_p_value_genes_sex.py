from scipy.stats import mannwhitneyu
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from utilz.Dataset import load_dataset
from utilz.helpers import plot_pca

meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Sex")

probs_to_drop = ds.y[ds.y == "n.a."].index

print(ds.y.value_counts())
ds.y = ds.y.drop(index=probs_to_drop)
ds.X = ds.X.drop(index=probs_to_drop)
print(ds.y.value_counts())

y = ds.y

gene_pvals = []
le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), index=y)

for gene in ds.X.columns:
    X = ds.X[gene]
    X_female = X[y == "F"]
    X_male = X[y == "M"]
    u_stat, p_val = mannwhitneyu(X_female, X_male, alternative="two-sided")

    if p_val < 1e-3:
        # cohen d
        mean_female = X_female.mean()
        mean_male = X_male.mean()
        n_f, n_m = len(X_female), len(X_male)
        pooled_std = np.sqrt(
            ((n_f - 1) * X_female.std() ** 2 + (n_m - 1) * X_male.std() ** 2)
            / (n_f + n_m - 2)
        )
        cohen_d = (mean_female - mean_male) / pooled_std
        if abs(cohen_d) > 0.8:
            print(f"Gene: {gene}, p-value: {p_val:.3e}, cohen's d: {cohen_d:.3f}")
            gene_pvals.append((gene, p_val))


gene_pvals.sort(key=lambda x: x[1])
sorted_genes = [gene for gene, pval in gene_pvals]
X_new = ds.X[sorted_genes].copy()
print(ds.X.shape)
print(X_new.shape)
print(X_new.head())
#X_new.T.to_csv(r"../../data/counts_pancreatic_filtered_sex.csv", sep=";", decimal=",")
num_pca_components = 5
plot_pca(X_new, y_encoded, num_pca_components, le)

