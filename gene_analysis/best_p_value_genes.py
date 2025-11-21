from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from Dataset import load_dataset
from utilz import *

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})
y = ds.y
gene_pvals = []
le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), index=y.index)

for gene in ds.X.columns:
    X = ds.X[gene]
    X_healthy = X[y == HEALTHY]
    X_cancer = X[y == CANCER]
    u_stat, p_val = mannwhitneyu(X_healthy, X_cancer, alternative="two-sided")
    auc = roc_auc_score(y_encoded, X)

    if p_val < 0.005:
        print(f"Gene: {gene}, p-value: {p_val:.3e}")
        gene_pvals.append((gene, p_val))


gene_pvals.sort(key=lambda x: x[1])
sorted_genes = [gene for gene, pval in gene_pvals]
X_new = ds.X[sorted_genes].copy()
print(ds.X.shape)
print(X_new.shape)
print(X_new.head())
X_new.T.to_csv(r"../../data/counts_pancreatic_filtered.csv", sep=";", decimal=",")

num_pca_components = 5
plot_pca(X_new, y_encoded, num_pca_components, le)