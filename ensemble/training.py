from collections import Counter

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.svm import SVC
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
#ds.y = ds.y.drop(ds.y[ds.y == DISEASE].index)

print(ds.y.value_counts())

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25))
sex_numeric = ds.sex.map({"F": 0, "M": 1})
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"Stosunek klas: {scale_pos_weight:.2f}")

logreg = LogisticRegression(max_iter=1500, class_weight='balanced',  fit_intercept=True)
model = SVC(
    kernel='rbf',
    C=8.6,
    gamma=0.00015,
    class_weight='balanced',
    probability=True,
    random_state=42
)

xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=300, colsample_bytree = 0.6,
                    gamma=0.14, learning_rate=0.12, max_depth=5, min_child_weight=3, reg_alpha=0.75,
                    reg_lambda=0.78, subsample=0.79, random_state=2137)

print("original X shape: ", X_train.shape)
preprocessing_pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaReductor', AnovaReductor(percentile=70)),
    ('MeanExpressionReductor', MeanExpressionReductor(percentile=30)),
    ('AgeBiasReductor',  CovariatesBiasReductor(covariate=ds.age)),
    ('SexBiasReductor',  CovariatesBiasReductor(covariate=sex_numeric)),
    ('scaler', StandardScaler())
])
voting = VotingClassifier(
    estimators=[('logreg', logreg), ('xgb', xgb), ('svc', model)],
    voting='soft',
    weights=[2, 2, 1]
)
pipeline = Pipeline([*preprocessing_pipeline.steps, ('ensemble', voting)])
pipeline.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_valid, pipeline.predict_proba(X_valid)[:, 1])
optimal_threshold = thresholds[np.argmax(tpr - fpr)]

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= optimal_threshold).astype(int)

show_report(y_pred, y_test, ds, le)
plot_roc_curve(y_proba, y_test, "ensemble")


print("Optimal threshold:", optimal_threshold)
print("AUC:", roc_auc_score(y_test, y_proba))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(len(le.classes_))))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))