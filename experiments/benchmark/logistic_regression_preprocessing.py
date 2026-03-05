from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import shap

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, y_train, y_test = train_test_split(ds.X, y_encoded, test_size=0.5,
                                                    random_state=42, stratify=y_encoded)

model = LogisticRegression(max_iter=1500, class_weight='balanced',  fit_intercept=True)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('NoneInformativeGeneReductor', ConstantExpressionReductor()),
    ('AnovaReductor', AnovaReductor()),
    ('MeanExpressionReductor', MeanExpressionReductor(3)),
    ('AgeBiasReductor', AgeBiasReductor(age=ds.age)),
    ('SexBiasReductor', SexBiasReductor(sex=ds.sex)),
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

plot_roc_curve(y_proba, y_test, "logistic regression")