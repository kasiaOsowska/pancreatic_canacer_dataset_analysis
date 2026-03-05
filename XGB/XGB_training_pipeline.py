from collections import Counter

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    balanced_accuracy_score
from sklearn.model_selection import train_test_split

from utilz.Dataset import load_dataset
from utilz.helpers import *
from utilz.preprocessing_utilz import *
from utilz.constans import DISEASE, HEALTHY


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, X_valid, y_train, y_test, y_valid = ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25)

class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"Stosunek klas: {scale_pos_weight:.2f}")
bst = XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=267, colsample_bytree= 0.905, gamma= 1.64, reg_alpha=1.81,
                    learning_rate=0.072, max_depth= 2, min_child_weight= 1, reg_lambda= 1.978, subsample= 0.53)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('constant',  ConstantExpressionReductor()),
    ('high_disp', HighDispersionReductor()),
    ('mean_expr', MeanExpressionReductor(4)),
    ('age_bias',  CovariatesBiasReductor(covariate=ds.age)),
    ('sex_bias',  CovariatesBiasReductor(covariate=ds.sex)),
    ('anova',     AnovaReductor()),
    ('scaler',    StandardScaler()),
    ('model', bst)
])

pipeline.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_valid, pipeline.predict_proba(X_valid)[:, 1])
optimal_threshold = thresholds[np.argmax(tpr - fpr)]

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= optimal_threshold).astype(int)
show_report(y_pred, y_test, ds, le)
plot_roc_curve(y_proba, y_test, "logistic regression")

show_report(y_pred, y_test, ds, le)
print("Optimal threshold:", optimal_threshold)
print("AUC:", roc_auc_score(y_test, y_proba))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(len(le.classes_))))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

