from Dataset import load_dataset
from utilz import *

meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
X = ds.X
meta = ds.meta
y = ds.y

print(X.shape, meta.shape)
print(y.value_counts())

sample_id = ds.X.sample(1).index[0]
print(ds.X.loc[sample_id])
print(ds.meta.loc[sample_id])

healthy = ds.X[ds.y == HEALTHY]
disease = ds.X[ds.y == DISEASE]
cancer = ds.X[ds.y == CANCER]

print("healthy stats")
print(healthy.values.mean())
print(healthy.values.std())
print(healthy.values.min())
print(healthy.values.max())
print("disease stats")
print(disease.values.mean())
print(disease.values.std())
print(disease.values.min())
print(disease.values.max())
print("cancer stats")
print(cancer.values.mean())
print(cancer.values.std())
print(cancer.values.min())
print(cancer.values.max())

X_train, y_train, X_test, y_test = ds.training_split(test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(y_train.value_counts())
print(y_test.value_counts())

sample_id = X_train.sample(1).index[0]
print(y_train.loc[sample_id])
print(X_train.loc[sample_id])