import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, balanced_accuracy_score, roc_auc_score, roc_curve)
from xgboost import XGBClassifier
from utilz.Dataset import load_dataset
from utilz.helpers import *
from utilz.preprocessing_utilz import *
from utilz.constans import DISEASE, HEALTHY

meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds_pathways = ds.get_pathway_scores(
    library='Reactome_2022',
    cache_dir='./cache/reactome'
)

X = ds_pathways.X.values
y = ds_pathways.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=2137, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=2137, stratify=y_temp
)

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

bst = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    colsample_bytree=0.9,
    gamma=1,
    reg_alpha=2,
    learning_rate=0.07,
    max_depth=2,
    min_child_weight=1,
    reg_lambda=2,
    subsample=0.5
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=50)),
    ('clf', bst)
])

pipeline.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_valid, pipeline.predict_proba(X_valid)[:, 1])
optimal_threshold = thresholds[np.argmax(tpr - fpr)]

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= optimal_threshold).astype(int)

print("Optimal threshold:", optimal_threshold)
print("AUC:", roc_auc_score(y_test, y_proba))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

plot_roc_curve(y_proba, y_test, "XGBoost + ssGSEA Reactome")