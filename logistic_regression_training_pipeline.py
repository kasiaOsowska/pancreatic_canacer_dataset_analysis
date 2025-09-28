import utilz
from Dataset import load_dataset

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import svm
from utilz import *


meta_path = r"../data/samples_pancreatic_filtered.xlsx"
data_path = r"../data/counts_pancreatic_filtered.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

X_train, y_train, X_test, y_test = ds.training_split(test_size=0.5, random_state=42)

# combine healthy and disease into one class
y_train = y_train.replace({DISEASE: HEALTHY})
y_test = y_test.replace({DISEASE: HEALTHY})


"""
test_to_drop = y_test[y_test == DISEASE].index
train_to_drop = y_train[y_train == DISEASE].index

y_test = y_test.drop(index=test_to_drop)
X_test = X_test.drop(index=test_to_drop)

y_train = y_train.drop(index=train_to_drop)
X_train = X_train.drop(index=train_to_drop)
"""

print("X train, y train shapes:")
print(X_train.shape, y_train.shape)
print("X test, y test shapes:")
print(X_test.shape, y_test.shape)


le = LabelEncoder()
y_train_encoded = pd.Series(le.fit_transform(y_train), index=y_train.index)
y_test_encoded = pd.Series(le.transform(y_test), index=y_test.index)

model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                           intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
                           max_iter=500, multi_class='deprecated', verbose=0, warm_start=False,
                           n_jobs=None, l1_ratio=None)

scaler = StandardScaler()
pca = PCA(n_components=1, svd_solver='full')
pipeline = Pipeline([('scaler', scaler), ('pca', pca), ('model', model)])


y_train_pred = pipeline.fit(X_train, y_train_encoded)
y_pred = pipeline.predict(X_test)
print("y test encoded:")
print(y_test_encoded)
print("y_pred")
utilz.save_report(y_pred, y_test_encoded, ds, le)

print(confusion_matrix(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))