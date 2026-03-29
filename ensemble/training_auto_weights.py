from collections import Counter
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25))

sex_numeric = ds.sex.map({"F": 0, "M": 1})
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"Stosunek klas: {scale_pos_weight:.2f}")

logreg = LogisticRegression(
    max_iter=1500, class_weight='balanced', fit_intercept=True
)
svc = SVC(
    kernel='rbf', C=8.6, gamma=0.00015,
    class_weight='balanced', probability=True, random_state=42
)
xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight, n_estimators=300,
    colsample_bytree=0.6, gamma=0.14, learning_rate=0.12,
    max_depth=5, min_child_weight=3, reg_alpha=0.75,
    reg_lambda=0.78, subsample=0.79, random_state=2137
)

print("original X shape:", X_train.shape)
preprocessing_pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaReductor', AnovaReductor(percentile=70)),
    ('MeanExpressionReductor', MeanExpressionReductor(percentile=30)),
    ('AgeBiasReductor', CovariatesBiasReductor(covariate=ds.age)),
    ('SexBiasReductor', CovariatesBiasReductor(covariate=sex_numeric)),
    ('scaler', StandardScaler())
])

preprocessing_pipeline.fit(X_train, y_train)
X_train_pp = preprocessing_pipeline.transform(X_train)
X_valid_pp = preprocessing_pipeline.transform(X_valid)
X_test_pp  = preprocessing_pipeline.transform(X_test)


stacking = StackingClassifier(
    estimators=[('logreg', logreg), ('xgb', xgb), ('svc', xgb)],
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
    cv=5,
    stack_method='predict_proba',
    passthrough=False
)
pipeline = Pipeline([*preprocessing_pipeline.steps, ('ensemble', stacking)])
pipeline.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_valid, pipeline.predict_proba(X_valid)[:, 1])
optimal_threshold = thresholds[np.argmax(tpr - fpr)]

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= optimal_threshold).astype(int)

show_report(y_pred, y_test, ds, le)
plot_roc_curve(y_proba, y_test, "ensemble (auto weights)")

print("\n" + "=" * 60)
print("WYNIKI – Ensemble z automatycznie dobranymi wagami")
print("=" * 60)
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"AUC:               {roc_auc_score(y_test, y_proba):.4f}")
print(f"F1 (weighted):     {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))}")
print(f"Classification report:\n"
      f"{classification_report(y_test, y_pred, target_names=le.classes_)}")
