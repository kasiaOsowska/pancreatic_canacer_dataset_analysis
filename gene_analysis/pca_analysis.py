from sklearn.preprocessing import LabelEncoder
from utilz import *
from Dataset import load_dataset

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic_filtered.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})
y = ds.y
meta = ds.meta
gene_pvals = []
le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), index=y.index)

num_pca_components = 5
plot_pca(ds.X, y_encoded, num_pca_components, le)