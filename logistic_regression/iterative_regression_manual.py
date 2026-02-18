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
        penalty='elasticnet', solver='saga', max_iter=15000,
        class_weight='balanced', l1_ratio=0.1, C=2, fit_intercept=True
    ).fit(X_train, y_train)

    y_pred = logreg.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average="weighted")
    print("F1 Score:", f1)

    explainer = shap.LinearExplainer(logreg, X_train)
    shap_values = explainer.shap_values(X_valid)
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

X_train, X_valid, y_train, y_valid = train_test_split(ds.X, y_encoded, test_size=0.5,
                                                    random_state=42, stratify=y_encoded)
X_test, X_valid, y_test, y_valid = train_test_split(X_valid, y_valid, test_size=0.5,
                                                    random_state=42, stratify=y_valid)

print("original X shape: ", X_train.shape)
preprocessing_pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('AnovaReductor', AnovaReductor()),
    ('MeanExpressionReductor', MeanExpressionReductor(3)),
    ('scaler', StandardScaler())
])
preprocessing_pipeline.set_output(transform="pandas")

X_train = preprocessing_pipeline.fit_transform(X_train, y_train)
X_valid = preprocessing_pipeline.transform(X_valid)

iteration = 0
f1 = 0
num_genes_to_drop = 500
while f1<0.9 and X_train.shape[1] > num_genes_to_drop:
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

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
show_report(y_pred, y_test, ds, le)
print("Macierz pomyłek:\n", cm)
print("\nRaport klasyfikacji:\n", classification_report(y_test, y_pred, target_names=le.classes_))