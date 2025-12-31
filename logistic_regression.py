from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
import shap

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, y_train, y_test = train_test_split(ds.X, y_encoded, test_size=0.5,
                                                    random_state=42, stratify=y_encoded)

model = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=1500,
    class_weight='balanced', l1_ratio = 0.8, C = 2
)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('VarianceExpressionReductor', VarianceExpressionReductor(0.1)),
    ('MeanExpressionReductor', MeanExpressionReductor(4)),
    ('PValueReductor', PValueReductor(0.005)),
    ('scaler', StandardScaler()),
    ('model', model)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.show()
show_report(y_pred, y_test, ds, le)

print("Confusion Matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

y_proba = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"AUC Score: {auc_score:.4f}")

indices_perfect_sensitivity = np.where(tpr == 1.0)[0]

if len(indices_perfect_sensitivity) > 0:
    best_idx = indices_perfect_sensitivity[np.argmin(fpr[indices_perfect_sensitivity])]

    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    best_threshold = thresholds[best_idx]
    best_specificity = 1 - best_fpr

    print(f"\n{'=' * 60}")
    print(f"Wyniki dla Sensitivity = 100%:")
    print(f"{'=' * 60}")
    print(f"Sensitivity (TPR):        {best_tpr * 100:.2f}%")
    print(f"Specificity (TNR):        {best_specificity * 100:.2f}%")
    print(f"False Positive Rate:      {best_fpr * 100:.2f}%")
    print(f"threshold:      {best_threshold:.4f}")
    print(f"{'=' * 60}")
else:
    print("Nie można osiągnąć 100% sensitivity z tym modelem.")
    closest_idx = np.argmax(tpr)
    print(f"Maksymalna osiągnięta sensitivity: {tpr[closest_idx] * 100:.2f}%")
    print(f"Odpowiadająca specificity: {(1 - fpr[closest_idx]) * 100:.2f}%")


logreg = pipeline.named_steps['model']

coefs = logreg.coef_[0]
intercept = logreg.intercept_[0]
preproc = pipeline[:-1]

feature_names = preproc.get_feature_names_out()
X_train_trans = preproc.transform(X_train)
X_test_trans  = preproc.transform(X_test)
X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names)
X_test_trans_df  = pd.DataFrame(X_test_trans,  columns=feature_names)

coef_series = (
    pd.Series(coefs, index=feature_names)
    .sort_values(ascending=False)
)

coef_series.to_csv("feature_weights.csv", header=["weight"])

explainer = shap.LinearExplainer(
    logreg,
    X_train_trans_df,
    feature_perturbation="interventional"
)

shap_values = explainer.shap_values(X_test_trans_df)

shap_values_series = pd.DataFrame(
    shap_values,
    columns=feature_names
).mean().sort_values(ascending=False)

print(shap_values_series.head(20))

shap_values_series = pd.DataFrame(
    shap_values,
    columns=feature_names
).mean().sort_values(ascending=True)

print(shap_values_series.head(20))