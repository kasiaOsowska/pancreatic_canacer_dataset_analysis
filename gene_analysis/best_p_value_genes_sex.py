from scipy.stats import mannwhitneyu
from sklearn.preprocessing import LabelEncoder
from utilz import *
from Dataset import load_dataset

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Sex")
meta = ds.meta

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

    if p_val < 1e-6:
        print(f"Gene: {gene}, p-value: {p_val:.3e}")
        gene_pvals.append((gene, p_val))


gene_pvals.sort(key=lambda x: x[1])
sorted_genes = [gene for gene, pval in gene_pvals]
X_new = ds.X[sorted_genes].copy()
print(ds.X.shape)
print(X_new.shape)
print(X_new.head())
X_new = X_new.T
X_new.to_csv(r"../../data/counts_pancreatic_filtered_sex.csv", sep=";", decimal=",")
meta.to_excel(r"../../data/samples_pancreatic_filtered_sex.xlsx")
num_pca_components = 2
plot_pca(X_new, y_encoded, num_pca_components, le)

