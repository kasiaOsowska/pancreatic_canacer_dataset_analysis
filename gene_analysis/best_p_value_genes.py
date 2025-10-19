from sklearn.decomposition import PCA

from Dataset import load_dataset
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from utilz import *
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})
y = ds.y
meta = ds.meta
gene_pvals = []
le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), index=y.index)

for gene in ds.X.columns:
    X = ds.X[gene]
    X_healthy = X[y == HEALTHY]
    X_cancer = X[y == CANCER]
    u_stat, p_val = mannwhitneyu(X_healthy, X_cancer, alternative="two-sided")
    auc = roc_auc_score(y_encoded, X)

    if p_val < 1e-6 and abs(0.5-auc) > 0.1:
        print(f"Gene: {gene}, p-value: {p_val:.3e}")
        gene_pvals.append((gene, p_val))


gene_pvals.sort(key=lambda x: x[1])
sorted_genes = [gene for gene, pval in gene_pvals]
X_new = ds.X[sorted_genes].copy()
print(ds.X.shape)
print(X_new.shape)
print(X_new.head())
X_new_T = X_new.T
X_new_T.to_csv(r"../../data/counts_pancreatic_filtered.csv", sep=";", decimal=",")
meta.to_excel(r"../../data/samples_pancreatic_filtered.xlsx")

viz_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(n_components=2, svd_solver="full", random_state=42))
])

X_train_pca = viz_pipe.fit_transform(X_new)
evr = viz_pipe.named_steps["pca"].explained_variance_ratio_
classes = le.classes_
y_test_int = y_encoded.values
cmap = plt.get_cmap("tab10")

plt.figure(figsize=(7,6))
for k, cls in enumerate(classes):
    mask = (y_test_int == k)
    plt.scatter(
        X_train_pca[mask, 0], X_train_pca[mask, 1],
        label=str(cls), alpha=0.75, s=45
    )

plt.xlabel(f"PC1 ({evr[0]*100:.1f}% wariancji)")
plt.ylabel(f"PC2 ({evr[1]*100:.1f}% wariancji)")
plt.title("PCA")
plt.legend(title="Klasa")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()




