import utilz
from Dataset import load_dataset

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

bst = XGBClassifier(n_estimators=220, max_depth=3, learning_rate=0.06, objective='binary:logistic',
                    colsample_bytree = 0.8, reg_lambda = 3.0, gamma = 0, min_child_weight = 1,
                    subsample = 0.9)
scaler = StandardScaler()
pipeline = Pipeline([('bst', bst)])


y_train_pred = pipeline.fit(X_train, y_train_encoded)
y_pred = pipeline.predict(X_test)
print("y test encoded:")
print(y_test_encoded)
print("y_pred")
utilz.save_report(y_pred, y_test_encoded, ds, le)

print(confusion_matrix(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))