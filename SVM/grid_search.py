from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import loguniform
import pandas as pd
import numpy as np
import shap

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

ds.y = ds.y.replace({DISEASE: HEALTHY})
print(ds.y.value_counts())

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(
        ds.X, y_encoded, test_size=0.25, valid_size=0.25
    )
)
sex_numeric = ds.sex.map({"F": 0, "M": 1})
base_pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaReductor',              AnovaReductor(percentile=80)),
    ('MeanExpressionReductor',     MeanExpressionReductor(percentile=25)),
    ('AgeBiasReductor',            CovariatesBiasReductor(covariate=ds.age)),
    ('SexBiasReductor',            CovariatesBiasReductor(covariate=sex_numeric)),
    ('scaler',                     StandardScaler()),
    ('model',                      SVC(probability=True, random_state=42)),
])

_preproc = {
    "AnovaReductor__percentile":          [60, 70, 75, 80, 85, 90],
    "MeanExpressionReductor__percentile": [10, 15, 20, 25, 30, 35],
}

param_dist = [
    {
        **_preproc,
        "model__kernel":       ["rbf"],
        "model__C":            loguniform(1e-2, 1e3),
        "model__gamma":        loguniform(1e-5, 1e-1),
        "model__class_weight": ["balanced", None],
    },
    {
        **_preproc,
        "model__kernel":       ["rbf"],
        "model__C":            loguniform(1e-2, 1e3),
        "model__gamma":        ["scale", "auto"],
        "model__class_weight": ["balanced", None],
    },
    {
        **_preproc,
        "model__kernel":       ["sigmoid"],
        "model__C":            loguniform(1e-2, 1e2),
        "model__gamma":        loguniform(1e-5, 1e-1),
        "model__coef0":        [-1.0, 0.0, 0.5, 1.0],
        "model__class_weight": ["balanced", None],
    },
]

cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_pipeline,
    param_distributions=param_dist,
    n_iter=80,
    scoring="average_precision",
    cv=cv_inner,
    n_jobs=-1,
    random_state=42,
    verbose=2,
    refit=True,
    return_train_score=True,
    error_score="raise",
)

search.fit(X_train, y_train)

print("\n=== BEST PARAMS ===")
print(search.best_params_)
print(f"Best CV balanced_accuracy: {search.best_score_:.4f}")

best_pipe = search.best_estimator_

fpr, tpr, thresholds = roc_curve(
    y_valid, best_pipe.predict_proba(X_valid)[:, 1]
)
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
print(f"\nOptimal threshold (Youden): {optimal_threshold:.4f}")

y_proba = best_pipe.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= optimal_threshold).astype(int)

show_report(y_pred, y_test, ds, le)
plot_roc_curve(y_proba, y_test, "SVC (tuned)")

print(f"AUC:               {roc_auc_score(y_test, y_proba):.4f}")
print(f"F1 (weighted):     {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print("Confusion matrix:\n",
      confusion_matrix(y_test, y_pred, labels=range(len(le.classes_))))
print("Classification report:\n",
      classification_report(y_test, y_pred, target_names=le.classes_))

"""
ROC AUC = 0.839
AUC:               0.8385
F1 (weighted):     0.8107
Balanced accuracy: 0.7870
{'AnovaReductor__percentile': 70, 'MeanExpressionReductor__percentile': 30, 'model__C': np.float64(8.585306974480472), 'model__class_weight': 'balanced', 'model__gamma': np.float64(0.000132965214572995), 'model__kernel': 'rbf'}

"""