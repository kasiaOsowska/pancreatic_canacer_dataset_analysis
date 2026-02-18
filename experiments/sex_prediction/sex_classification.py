from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utilz.preprocessing_utilz import *
from utilz.helpers import *
from utilz.Dataset import load_dataset


meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.dropna()

y = ds
y = y.dropna().astype(int)
# Prediction of age is only meaningful for healthy samples
healthy_idx = ds.y[ds.y == HEALTHY].index
idx = y.index.intersection(healthy_idx)
X_train, X_test, y_train, y_test = train_test_split(ds.X, ds.y, test_size=0.5,
                                                    random_state=42, stratify=ds.y)

print("X train, y train shapes:")
print(X_train.shape, y_train.shape)
print("X test, y test shapes:")
print(X_test.shape, y_test.shape)


le = LabelEncoder()
y_train_encoded = pd.Series(le.fit_transform(y_train), index=y_train.index)
y_test_encoded = pd.Series(le.transform(y_test), index=y_test.index)

model = LogisticRegression(
    penalty='l2', solver='lbfgs', max_iter=500,
    class_weight='balanced',
)

scaler = StandardScaler()
pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('scaler', StandardScaler()), ('model', model)
])

y_train_pred = pipeline.fit(X_train, y_train_encoded)
y_pred = pipeline.predict(X_test)
print("y test encoded:")
print(y_test_encoded)
print("y_pred")
show_report(y_pred, y_test_encoded, ds, le)

print(confusion_matrix(y_test_encoded, y_pred, labels=[0, 1]))
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))