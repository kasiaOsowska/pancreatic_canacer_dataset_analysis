from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.svm import SVC

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

print(ds.y.value_counts())

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.2, valid_size=0.2))
sex_numeric = ds.sex.map({"F": 0, "M": 1})
#{'AnovaReductor__percentile': 70, 'MeanExpressionReductor__percentile': 30, 'model__C': np.float64(8.585306974480472), 'model__class_weight': 'balanced', 'model__gamma': np.float64(0.000132965214572995), 'model__kernel': 'rbf'}

model = SVC(
    kernel='rbf',
    C=8.6,
    gamma=0.00015,
    class_weight='balanced',
    probability=True,
    random_state=42
)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaReductor', AnovaReductor(percentile=60)),
    ('MeanExpressionReductor', MeanExpressionReductor(percentile=20)),
    ('AgeBiasReductor',  CovariatesBiasReductor(covariate=ds.age)),
    ('SexBiasReductor',  CovariatesBiasReductor(covariate=sex_numeric)),
    ('scaler', StandardScaler()),
    ('model',model),
])

pipeline.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_valid, pipeline.predict_proba(X_valid)[:, 1])
optimal_threshold = thresholds[np.argmax(tpr - fpr)]

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= optimal_threshold).astype(int)

show_report(y_pred, y_test, ds, le)
plot_roc_curve(y_proba, y_test, "logistic regression")
print("Optimal threshold:", optimal_threshold)
print("AUC:", roc_auc_score(y_test, y_proba))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(len(le.classes_))))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
preproc = Pipeline(pipeline.steps[:-1])
feature_names   = preproc.get_feature_names_out()
X_train_trans   = pd.DataFrame(preproc.transform(X_train), columns=feature_names)
X_test_trans    = pd.DataFrame(preproc.transform(X_test),  columns=feature_names)