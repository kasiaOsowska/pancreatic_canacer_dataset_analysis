from Dataset import load_dataset

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from utilz import *


meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
X = ds.X
meta = ds.meta
y = ds.y
y = y.replace({DISEASE: HEALTHY})


print(X.shape)

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), index=y.index)


threshold_mean = 2
mean_per_gene = X.mean(axis=0)
high_mean_genes = mean_per_gene[mean_per_gene > threshold_mean].index
X = X[high_mean_genes]


threshold_variance = 0.1
variance_per_gene = X.var(axis=0)
high_variance_genes = variance_per_gene[variance_per_gene > threshold_variance].index
X = X[high_variance_genes]

""""
info_gain = mutual_info_classif(X, y_encoded, discrete_features=False, random_state=42, n_neighbors=5)
gene_scores = pd.Series(info_gain, index=X.columns)
gene_scores = gene_scores.sort_values(ascending=False)
X = X[gene_scores.head(200).index]
"""

print(X.sample(1))
print(X.shape)


X = X.T
X.to_csv(r"../data/counts_pancreatic_filtered.csv", sep=";", decimal=",")
meta.to_excel(r"../data/samples_pancreatic_filtered.xlsx")


