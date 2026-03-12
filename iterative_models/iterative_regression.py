from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, X_valid, y_train, y_test, y_valid = ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.2, valid_size=0.2)
#Najlepsze parametry: {'model__l1_ratio': 0.1, 'prep__MeanExpressionReductor__mean_threshold': 3, 'prep__PValueReductor__p_threshold': 0.1, 'prep__VarianceExpressionReductor__v_threshold': 0.2}

model = LogisticRegression(
    solver='saga', max_iter=15000,
    class_weight='balanced', l1_ratio = 0.1, C = 2, fit_intercept=True
)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('HighVarianceReductor', HighVarianceReductor(percentile=95)),
    ('mean_expr', MeanExpressionReductor(percentile=25)),
    ('AgeBiasReductor',  CovariatesBiasReductor(covariate=ds.age)),
    ("rfe", RFE(estimator=model, n_features_to_select=1500, step=0.2, verbose = 1)),
])

pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_valid)[:, 1]

fpr, tpr, thresholds = roc_curve(y_valid, y_proba)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold:", optimal_threshold)

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= optimal_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
auc_score = roc_auc_score(y_test, y_proba)
show_report(y_pred, y_test, ds, le)
print("Confusion Matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
plot_roc_curve(y_proba, y_test, "logistic regression")

print("f1 score: ", f1_score(y_test, y_pred, average="weighted"))