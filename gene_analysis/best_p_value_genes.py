from Dataset import load_dataset
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from utilz import *
from matplotlib import pyplot as plt

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y = ds.y
meta = ds.meta
gene_pvals = []
le = LabelEncoder()
y_combined = y.replace({DISEASE: HEALTHY})
y_encoded = pd.Series(le.fit_transform(y_combined), index=y_combined)

for gene in ds.X.columns:
    X = ds.X[gene]
    X_healthy = X[y.isin([HEALTHY, DISEASE])]
    X_cancer = X[y == CANCER]
    t_stat, p_val = mannwhitneyu(X_healthy, X_cancer, alternative="two-sided")
    auc = roc_auc_score(y_encoded, X)

    if p_val < 1e-6 and auc > 0.5:
        print(f"Gene: {gene}, p-value: {p_val:.3e}")
        gene_pvals.append((gene, p_val))


gene_pvals.sort(key=lambda x: x[1])
sorted_genes = [gene for gene, pval in gene_pvals]
X_new = ds.X[sorted_genes].copy()
print(ds.X.shape)
print(X_new.shape)
print(X_new.head())
X_new = X_new.T
X_new.to_csv(r"../../data/counts_pancreatic_filtered.csv", sep=";", decimal=",")
meta.to_excel(r"../../data/samples_pancreatic_filtered.xlsx")
