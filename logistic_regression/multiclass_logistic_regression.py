from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.multiclass import OneVsRestClassifier

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group", separate_stage_iv=True)
ds.y = ds.y.replace({DISEASE: HEALTHY})

# y zawiera teraz: healthy, cancer, cancer_IV

le = LabelEncoder()
print(ds.y.value_counts())
ds.y = ds.y.drop(ds.y[ds.y == "cancer_IV"].index)
print(ds.y.value_counts())

y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.3, valid_size=0.3))
sex_numeric = ds.sex.map({"F": 0, "M": 1})
print(y_test.value_counts())
print(y_valid.value_counts())
print(y_train.value_counts())
model = LogisticRegression(max_iter=1500, class_weight='balanced',  fit_intercept=True)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaReductor', AnovaReductor(percentile=95)),
    ('MeanExpressionReductor', MeanExpressionReductor(percentile=25)),
    ('AgeBiasReductor', CovariatesBiasReductor(covariate=ds.age)),
    ('SexBiasReductor', CovariatesBiasReductor(covariate=sex_numeric)),
    ('scaler', StandardScaler()),
    ('model', model)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)  # ← bezpośrednio klasy, nie progi

print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(len(le.classes_))))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))