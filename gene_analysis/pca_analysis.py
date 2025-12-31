from sklearn.preprocessing import LabelEncoder
import pandas as pd

from utilz.Dataset import load_dataset
from utilz.helpers import plot_pca
from utilz.constans import DISEASE, HEALTHY

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic_filtered.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})
meta = ds.meta

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X = ds.X.iloc[:, :10]
num_pca_components = 5
plot_pca(X, y_encoded, num_pca_components, le)