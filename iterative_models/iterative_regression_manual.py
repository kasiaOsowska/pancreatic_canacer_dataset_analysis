from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix


def fit_iterative_logistic_regression(X_train, y_train, X_valid, y_valid, num_genes_to_drop):

    logreg = LogisticRegression(
        solver='saga', max_iter=15000,
        class_weight='balanced', l1_ratio=0.1, C=2, fit_intercept=True
    ).fit(X_train, y_train)

    y_pred = logreg.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average="weighted")
    print("F1 Score:", f1)

    explainer = shap.LinearExplainer(logreg, X_train)
    shap_values = explainer.shap_values(X_train)
    # shap_values: (n_samples, n_features)
    shap_per_feature = np.mean(abs(shap_values), axis=0)

    drop_idx = np.argsort(shap_per_feature)[:num_genes_to_drop]
    drop_cols = X_train.columns[drop_idx]
    X_train_new = X_train.drop(columns=drop_cols)
    X_valid_new = X_valid.drop(columns=drop_cols)

    return X_train_new, X_valid_new, f1


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, X_valid, y_train, y_test, y_valid = ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25)

print("original X shape: ", X_train.shape)
preprocessing_pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('HighVarianceReductor', AnovaReductor(percentile=95)),
    ('mean_expr', MeanExpressionReductor(percentile=25)),
    #('AgeBiasReductor',  CovariatesBiasReductor(covariate=ds.age)),
    ('scaler',                     StandardScaler()),
])
preprocessing_pipeline.set_output(transform="pandas")

X_train = preprocessing_pipeline.fit_transform(X_train, y_train)
X_valid = preprocessing_pipeline.transform(X_valid)

iteration = 0
f1 = 0
num_genes_to_drop = 200
while iteration<19 and X_train.shape[1] > num_genes_to_drop:
    print(f"Iteration {iteration}")
    X_train, X_valid, f1 = fit_iterative_logistic_regression(X_train, y_train,
                                                            X_valid, y_valid, num_genes_to_drop)
    iteration += 1


X_test = preprocessing_pipeline.transform(X_test)
kept_cols = X_train.columns
X_test = X_test.reindex(columns=kept_cols)

logreg = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=15000,
    class_weight='balanced', l1_ratio=0.1, C=2, fit_intercept=True
).fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_valid, logreg.predict_proba(X_valid)[:, 1])
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
print("Optimal threshold:", optimal_threshold)

y_proba = logreg.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= optimal_threshold).astype(int)

print("AUC:", roc_auc_score(y_test, y_proba))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(len(le.classes_))))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

show_report(y_pred, y_test, ds, le)
plot_roc_curve(y_proba, y_test, "logistic regression")