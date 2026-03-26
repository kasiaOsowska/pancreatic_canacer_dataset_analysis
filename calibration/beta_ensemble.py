from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
from netcal.scaling import BetaCalibration
from netcal.metrics import ECE

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

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25))
sex_numeric = ds.sex.map({"F": 0, "M": 1})
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"num non-cancer / num cancer: {scale_pos_weight:.2f}")

logreg = LogisticRegression(max_iter=1500, class_weight='balanced',  fit_intercept=True, random_state=2137)
model = SVC(
    kernel='rbf',
    C=8.6,
    gamma=0.00015,
    class_weight='balanced',
    probability=True,
    random_state=2137
)

xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=300, colsample_bytree = 0.6,
                    gamma=0.14, learning_rate=0.12, max_depth=5, min_child_weight=3, reg_alpha=0.75,
                    reg_lambda=0.78, subsample=0.79, random_state=2137)

print("original X shape: ", X_train.shape)
preprocessing_pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaReductor', AnovaReductor(percentile=70)),
    ('MeanExpressionReductor', MeanExpressionReductor(percentile=10)),
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

# ── Wyniki PRZED kalibracją ───────────────────────────────────────────────────
y_proba = pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_valid, pipeline.predict_proba(X_valid)[:, 1])
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
y_pred_raw  = (y_proba >= optimal_threshold).astype(int)
print("=== BEZ KALIBRACJI ===")
print("F1 (weighted):     ", f1_score(y_test, y_pred_raw, average="weighted"))
print("Balanced accuracy: ", balanced_accuracy_score(y_test, y_pred_raw))
print(classification_report(y_test, y_pred_raw, target_names=le.classes_))

# ── Beta Calibration (netcal) ─────────────────────────────────────────────────
scores_cal  = pipeline.predict_proba(X_valid)[:, 1]   # netcal przyjmuje 1D
scores_test = pipeline.predict_proba(X_test)[:, 1]

cal = BetaCalibration(detection=False)
cal.fit(scores_cal, y_valid.values)

p_calibrated = cal.transform(scores_test)   # 1D array, p(cancer)
y_pred_cal   = (p_calibrated >= 0.5).astype(int)

# ── Wyniki PO kalibracji ──────────────────────────────────────────────────────
print("\n=== PO KALIBRACJI (Beta Calibration) ===")
print("F1 (weighted):     ", f1_score(y_test, y_pred_cal, average="weighted"))
print("Balanced accuracy: ", balanced_accuracy_score(y_test, y_pred_cal))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_cal, labels=range(len(le.classes_))))
print(classification_report(y_test, y_pred_cal, target_names=le.classes_))

# ── Metryki proper scoring ────────────────────────────────────────────────────
ece = ECE(10)
print(f"\nLog-loss  | Raw: {log_loss(y_test, scores_test):.4f}        | Cal: {log_loss(y_test, p_calibrated):.4f}")
print(f"Brier     | Raw: {brier_score_loss(y_test, scores_test):.4f}  | Cal: {brier_score_loss(y_test, p_calibrated):.4f}")
print(f"ECE       | Raw: {ece.measure(scores_test, y_test.values):.4f} | Cal: {ece.measure(p_calibrated, y_test.values):.4f}")

# ── Wykresy ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for probs, label, color in [
    (scores_test,  "Raw LR",           "steelblue"),
    (p_calibrated, "Beta Calibration", "darkorange"),
]:
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    axes[0].plot(mean_pred, frac_pos, marker='o', label=label, color=color)

axes[0].plot([0, 1], [0, 1], 'k--', label="Perfect")
axes[0].set_title("Reliability diagram")
axes[0].set_xlabel("Predicted probability")
axes[0].set_ylabel("True fraction of positives")
axes[0].legend()

axes[1].hist(scores_test,  bins=20, alpha=0.5, label="Raw LR",           color="steelblue")
axes[1].hist(p_calibrated, bins=20, alpha=0.5, label="Beta Calibration", color="darkorange")
axes[1].set_title("Rozkład prawdopodobieństw")
axes[1].set_xlabel("p(cancer)")
axes[1].legend()

plt.tight_layout()
plt.savefig("calibration_comparison.png", dpi=150)
plt.show()
