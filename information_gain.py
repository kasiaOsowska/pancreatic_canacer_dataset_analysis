from Dataset import load_dataset

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt


meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
X = ds.X
meta = ds.meta
y = ds.y

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), index=y.index)

std_per_gene = X.std(axis=0)
plt.figure(figsize=(8, 5))
plt.yscale("log")
plt.hist(std_per_gene, bins=50, edgecolor='black')
plt.xlabel("Odchylenie standardowe genu")
plt.ylabel("Liczba genów")
plt.title("Rozkład odchyleń standardowych genów")
plt.tight_layout()
plt.savefig("graphics/std_per_gene.png")
plt.show()

threshold = 1.5
genes_low_std = std_per_gene[std_per_gene < threshold].index
X_filtered_by_std = X[genes_low_std]
std_per_gene_cut = X_filtered_by_std.std(axis=0)
plt.figure(figsize=(8, 5))
plt.yscale("log")
plt.hist(std_per_gene_cut, bins=50, edgecolor='black')
plt.xlabel("Odchylenie standardowe genu")
plt.ylabel("Liczba genów")
plt.title("Rozkład odchyleń standardowych genów po przycięciu "+ str(threshold))
plt.tight_layout()
plt.savefig("graphics/std_per_gene_cut.png")
plt.show()

mean_per_gene = X.mean(axis=0)
plt.figure(figsize=(8, 5))
plt.hist(mean_per_gene, bins=50, edgecolor='black')
plt.yscale("log")
plt.xlabel("Średnia wartość genu")
plt.ylabel("Liczba genów")
plt.title("Rozkład średnich wartości genów")
plt.tight_layout()
plt.savefig("graphics/mean_per_gene.png")
plt.show()

variance_per_gene = X.var(axis=0)
plt.figure(figsize=(8, 5))
plt.hist(variance_per_gene, bins=50, edgecolor='black')
plt.yscale("log")
plt.xlabel("Wariancja genu")
plt.ylabel("Liczba genów")
plt.title("Rozkład wariancji genów")
plt.tight_layout()
plt.savefig("graphics/variance_per_gene.png")
plt.show()

"""

info_gain = mutual_info_classif(ds.X, y_encoded, discrete_features=False, random_state=42, n_neighbors=3)

gene_scores = pd.Series(info_gain, index=ds.X.columns)
gene_scores = gene_scores.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
plt.hist(info_gain, bins=50, edgecolor='black')
plt.yscale("log")
plt.xlabel("Informacja wzajemna")
plt.ylabel("Liczba genów")
plt.title("Rozkład informacji wzajemnej genów")
plt.tight_layout()
plt.savefig("graphics/information_gain.png")
plt.show()

info_gain = mutual_info_classif(ds.X, y_encoded, discrete_features=False, random_state=42, n_neighbors=6)

gene_scores = pd.Series(info_gain, index=ds.X.columns)
gene_scores = gene_scores.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
plt.hist(info_gain, bins=50, edgecolor='black')
plt.yscale("log")
plt.xlabel("Informacja wzajemna")
plt.ylabel("Liczba genów")
plt.title("Rozkład informacji wzajemnej genów")
plt.tight_layout()
plt.savefig("graphics/information_gain_6.png")
plt.show()

info_gain = mutual_info_classif(ds.X, y_encoded, discrete_features=False, random_state=42, n_neighbors=12)

gene_scores = pd.Series(info_gain, index=ds.X.columns)
gene_scores = gene_scores.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
plt.hist(info_gain, bins=50, edgecolor='black')
plt.yscale("log")
plt.xlabel("Informacja wzajemna")
plt.ylabel("Liczba genów")
plt.title("Rozkład informacji wzajemnej genów")
plt.tight_layout()
plt.savefig("graphics/information_gain_12.png")
plt.show()


info_gain = mutual_info_classif(ds.X, y_encoded, discrete_features=False, random_state=42, n_neighbors=20)

gene_scores = pd.Series(info_gain, index=ds.X.columns)
gene_scores = gene_scores.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
plt.hist(info_gain, bins=50, edgecolor='black')
plt.yscale("log")
plt.xlabel("Informacja wzajemna")
plt.ylabel("Liczba genów")
plt.title("Rozkład informacji wzajemnej genów")
plt.tight_layout()
plt.savefig("graphics/information_gain_20.png")
plt.show()

info_gain = mutual_info_classif(ds.X, y_encoded, discrete_features=False, random_state=42, n_neighbors=30)

gene_scores = pd.Series(info_gain, index=ds.X.columns)
gene_scores = gene_scores.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
plt.hist(info_gain, bins=50, edgecolor='black')
plt.yscale("log")
plt.xlabel("Informacja wzajemna")
plt.ylabel("Liczba genów")
plt.title("Rozkład informacji wzajemnej genów")
plt.tight_layout()
plt.savefig("graphics/information_gain_30.png")
plt.show()


threshold = 0.1
informative_genes = gene_scores[gene_scores > threshold].index
X_filtered = X[informative_genes]

info_gain_v2 = mutual_info_classif(X_filtered, y_encoded, discrete_features=False, random_state=42)

plt.figure(figsize=(8, 5))
plt.hist(info_gain_v2, bins=50, edgecolor='black')
plt.yscale("log")
plt.xlabel("Informacja wzajemna po przycięciu")
plt.ylabel("Liczba genów")
plt.title("Rozkład informacji wzajemnej genów")
plt.tight_layout()
plt.savefig("graphics/information_gain_cut.png")
plt.show()

"""

