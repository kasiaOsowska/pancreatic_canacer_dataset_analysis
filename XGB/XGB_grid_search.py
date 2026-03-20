from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from collections import Counter
import numpy as np
import shap

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *
from utilz.constans import DISEASE, HEALTHY
from xgboost import XGBClassifier

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

# Podział przed preprocessingiem
X_train, X_test, y_train, y_test = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.4, valid_size=0, return_valid = False))

# scale_pos_weight tylko z train
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]

pipe = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('HighVarianceReductor',       HighVarianceReductor(percentile=95)),
    ('mean_expr',                  MeanExpressionReductor(percentile=25)),
    ('AgeBiasReductor',            CovariatesBiasReductor(covariate=ds.age.loc[X_train.index])),
    ('scaler',                     StandardScaler()),
    ('clf',                        XGBClassifier(
                                        scale_pos_weight=scale_pos_weight,
                                        eval_metric='logloss',
                                        random_state=42,
                                        n_jobs=1,
                                    )),
])

param_dist = {
    'clf__n_estimators':     randint(100, 500),
    'clf__max_depth':        randint(2, 7),
    'clf__learning_rate':    uniform(0.01, 0.29),
    'clf__gamma':            uniform(0, 2),
    'clf__reg_lambda':       uniform(0.5, 3.0),
    'clf__reg_alpha':        uniform(0.0, 2.0),
    'clf__colsample_bytree': uniform(0.5, 0.5),
    'clf__subsample':        uniform(0.5, 0.5),
    'clf__min_child_weight': randint(1, 10),
}

X_train_valid = pd.concat([X_train, X_test])
y_train_valid = pd.concat([y_train, y_test])
train_idx = np.arange(len(X_train))
valid_idx = np.arange(len(X_train), len(X_train) + len(X_test))
fixed_split = [(train_idx, valid_idx)]

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=100,
    scoring='roc_auc',
    cv=fixed_split,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True,
)

search.fit(X_train_valid, y_train_valid)

print("Najlepsze parametry:", search.best_params_)
print("Najlepszy wynik na valid (AUC):", search.best_score_)

# Ewaluacja końcowa — tylko raz, na surowym X_test
y_test_pred  = search.predict(X_test)
y_test_proba = search.predict_proba(X_test)[:, 1]

print("\n=== TEST ===")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))

