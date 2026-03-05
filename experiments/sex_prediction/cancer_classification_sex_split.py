from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score,
                             balanced_accuracy_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from utilz.constans import DISEASE, HEALTHY
from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *

meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)


def make_pipeline(model, age_covariate):
    return Pipeline([
        ('constant',  ConstantExpressionReductor()),
        ('high_disp', HighDispersionReductor()),
        ('mean_expr', MeanExpressionReductor(3)),
        ('age_bias',  CovariatesBiasReductor(covariate=age_covariate)),
        ('anova',     AnovaReductor()),
        ('scaler',    StandardScaler()),
        ('model',     model),
    ])


def evaluate(pipeline, X_valid, y_valid, X_test, y_test):
    fpr, tpr, thresholds = roc_curve(y_valid, pipeline.predict_proba(X_valid)[:, 1])
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= optimal_threshold).astype(int)

    print("Optimal threshold:", optimal_threshold)
    print("AUC:", roc_auc_score(y_test, y_proba))
    print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    return y_pred


# FEMALE
X_f, y_f = ds.X.loc[ds.sex == 'F'], y_encoded.loc[ds.sex == 'F']
Xtr_f, Xte_f, Xva_f, ytr_f, yte_f, yva_f = ds.get_train_test_valid_split(X_f, y_f, test_size=0.25, valid_size=0.25)

pipe_f = make_pipeline(
    LogisticRegression(solver='saga', max_iter=15000,
                       class_weight='balanced', l1_ratio=0.8, C=2),
    ds.age)
pipe_f.fit(Xtr_f, ytr_f)

print("=== FEMALE ===")
pred_f = evaluate(pipe_f, Xva_f, yva_f, Xte_f, yte_f)

# MALE
X_m, y_m = ds.X.loc[ds.sex == 'M'], y_encoded.loc[ds.sex == 'M']
Xtr_m, Xte_m, Xva_m, ytr_m, yte_m, yva_m = ds.get_train_test_valid_split(X_m, y_m, test_size=0.25, valid_size=0.25)

pipe_m = make_pipeline(
    LogisticRegression(solver='saga', max_iter=1500,
                       class_weight='balanced', l1_ratio=0.8, C=2),
    ds.age)
pipe_m.fit(Xtr_m, ytr_m)

print("=== MALE ===")
pred_m = evaluate(pipe_m, Xva_m, yva_m, Xte_m, yte_m)

# AGGREGATE
print("=== AGGREGATED ===")
y_true_agg = pd.concat([yte_f, yte_m])
y_pred_agg = np.concatenate([pred_f, pred_m])

cm = confusion_matrix(y_true_agg, y_pred_agg, labels=range(len(le.classes_)))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(cmap="Blues", colorbar=False)
plt.title("Confusion matrix combined")
plt.show()
print(classification_report(y_true_agg, y_pred_agg, target_names=le.classes_))
print("F1 (weighted):", f1_score(y_true_agg, y_pred_agg, average="weighted"))
print("Balanced accuracy:", balanced_accuracy_score(y_true_agg, y_pred_agg))
