from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import shap

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
print(ds.X.shape)
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25))

model = LogisticRegression(
    solver='saga', max_iter=15000,
    class_weight='balanced', l1_ratio=0.1, C=2, fit_intercept=True
)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('HighVarianceReductor', HighVarianceReductor(percentile=95)),
    ('mean_expr', MeanExpressionReductor(percentile=25)),
    ('AgeBiasReductor',  CovariatesBiasReductor(covariate=ds.age)),
    ('scaler',                     StandardScaler()),
    ('model',                      model),
])

pipeline.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_valid, pipeline.predict_proba(X_valid)[:, 1])
optimal_threshold = thresholds[np.argmax(tpr - fpr)]

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= optimal_threshold).astype(int)

show_report(y_pred, y_test, ds, le)
plot_roc_curve(y_proba, y_test, "logistic regression")

preproc = Pipeline(pipeline.steps[:-1])
feature_names   = preproc.get_feature_names_out()
X_train_trans   = pd.DataFrame(preproc.transform(X_train), columns=feature_names)
X_test_trans    = pd.DataFrame(preproc.transform(X_test),  columns=feature_names)

pd.Series(pipeline.named_steps['model'].coef_[0], index=feature_names) \
    .sort_values(ascending=False) \
    .to_csv("feature_weights.csv", header=["weight"])

shap_values = shap.LinearExplainer(pipeline.named_steps['model'], X_train_trans) \
    .shap_values(X_test_trans)
pd.DataFrame(shap_values, columns=feature_names) \
    .mean() \
    .sort_values(ascending=False) \
    .to_csv("shap_values.csv", header=["weight"])

print("Optimal threshold:", optimal_threshold)
print("AUC:", roc_auc_score(y_test, y_proba))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(len(le.classes_))))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
