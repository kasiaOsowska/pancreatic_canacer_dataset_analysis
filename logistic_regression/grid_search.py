from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import (
    PredefinedSplit,
    RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, loguniform
import numpy as np
from scipy.stats import randint
from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *
from utilz.constans import DISEASE, HEALTHY


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)


X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25))
sex_numeric = ds.sex.map({"F": 0, "M": 1})

# Combine train + valid for RandomizedSearchCV with PredefinedSplit
X_search = pd.concat([X_train, X_valid])
y_search = pd.concat([y_train, y_valid])
# -1 = train (never in validation fold), 0 = validation fold
split_index = np.concatenate([np.full(len(X_train), -1), np.full(len(X_valid), 0)])
cv = PredefinedSplit(split_index)

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)
print("X_test shape: ", X_test.shape)

preprocessing_pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaReductor', AnovaReductor(percentile=80)),
    ('MeanExpressionReductor', MeanExpressionReductor(percentile=25)),
    ('AgeBiasReductor', CovariatesBiasReductor(covariate=ds.age)),
    ('SexBiasReductor', CovariatesBiasReductor(covariate=sex_numeric)),
    ('scaler', StandardScaler())
])

model = LogisticRegression(
    solver='saga',
    max_iter=15000,
    class_weight='balanced'
)

full_pipeline = Pipeline([
    ('prep', preprocessing_pipeline),
    ('model', model)
])

param_distributions = {
    'prep__AnovaReductor__percentile': uniform(50, 40),           # 50-90
    'prep__MeanExpressionReductor__percentile': uniform(5, 25),   # 5-30
    'prep__AgeBiasReductor__beta_thresh': uniform(0, 0.026),   # 0.002-0.028 (|β| max ~0.029, mean ~0.003)
    'prep__SexBiasReductor__beta_thresh': uniform(0, 0.35),    # 0.05-0.40 (|β| mean ~0.09, most genes < 0.4)
    'model__C': loguniform(0.01, 100.0),
    'model__l1_ratio': uniform(0, 1),
    'model__tol': loguniform(1e-5, 1e-3),
}

random_search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True,
    random_state=42,
    error_score='raise'
)

random_search.fit(X_search, y_search)

print("Najlepsze parametry:", random_search.best_params_)
print("Najlepszy wynik na valid (AUC):", random_search.best_score_)

y_test_pred  = random_search.predict(X_test)
y_test_proba = random_search.predict_proba(X_test)[:, 1]

print("\n=== TEST ===")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))

"""
Najlepsze parametry: {'model__C': 0.012637946338082878, 'model__l1_ratio': 0.10789142699330445, 'model__tol': 1.1557352816269867e-05, 'prep__AgeBiasReductor__beta_thresh': 0.01654667069285829, 'prep__AnovaReductor__percentile': 62.57423924305307, 'prep__MeanExpressionReductor__percentile': 17.71426727911757, 'prep__SexBiasReductor__beta_thresh': 0.31764826587413253}
Najlepszy wynik na valid (AUC): 0.8895946763460376
"""