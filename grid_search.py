from Dataset import load_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from utilz import *


meta_path = r"../data/samples_pancreatic_filtered.xlsx"
data_path = r"../data/counts_pancreatic_filtered.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

X_train, y_train, X_test, y_test = ds.training_split(test_size=0.2, random_state=42)
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

le = LabelEncoder()
y_train_encoded = pd.Series(le.fit_transform(y_train), index=y_train.index)
y_test_encoded = pd.Series(le.transform(y_test), index=y_test.index)
bst = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample = 0.9,
                    objective='binary:logistic')
scaler = StandardScaler()
pipeline = Pipeline([('bst', bst)])

param_grid = {
    'bst__colsample_bytree': [0.7, 0.8],
    'bst__reg_lambda': [2.0, 3.0],
    'bst__gamma': [0, 1],
    'bst__min_child_weight': [1, 2],
    'bst__n_estimators': [200, 220],
    'bst__max_depth': [3, 4],
    'bst__learning_rate': [0.04, 0.05, 0.055, 0.06],
    'bst__subsample': [0.8, 0.9]
}

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid.fit(X_train, y_train_encoded)
print("Najlepsze parametry:", grid.best_params_)
print("Najlepszy wynik CV (f1):", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(confusion_matrix(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))