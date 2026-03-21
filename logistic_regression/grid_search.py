from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, loguniform
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
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


X_train, X_test, y_train, y_test = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25, return_valid=False))
sex_numeric = ds.sex.map({"F": 0, "M": 1})

print("X_train shape:", X_train.shape)
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_distributions = {
    'prep__MeanExpressionReductor__percentile': uniform(10, 10),
    'prep__AnovaReductor__percentile': uniform(70, 10),
    'model__C': loguniform(0.1, 10.0),

    'model__l1_ratio': uniform(0.1, 0.8),
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

random_search.fit(X_train, y_train)

print("Najlepsze parametry:", random_search.best_params_)
print("Najlepszy wynik CV (f1):", random_search.best_score_)

print("Najlepsze parametry:", random_search.best_params_)
print("Najlepszy wynik na valid (AUC):", random_search.best_score_)

y_test_pred  = random_search.predict(X_test)
y_test_proba = random_search.predict_proba(X_test)[:, 1]

print("\n=== TEST ===")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))

"""
Najlepsze parametry: {'model__C': np.float64(1.2172847081122433), 'model__l1_ratio': np.float64(0.21273937997981013), 'model__tol': np.float64(0.0004021554526690286), 'prep__AnovaReductor__percentile': np.float64(70.74550643679771), 'prep__MeanExpressionReductor__percentile': np.float64(19.86886936600517)}
Najlepszy wynik na valid (AUC): 0.8116909253537161
"""