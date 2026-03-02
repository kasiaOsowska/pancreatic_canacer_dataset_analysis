from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X_train, X_test, y_train, y_test = train_test_split(ds.X, y_encoded, test_size=0.5,
                                                    random_state=42, stratify=y_encoded)
#Najlepsze parametry: {'model__l1_ratio': 0.1, 'prep__MeanExpressionReductor__mean_threshold': 3, 'prep__PValueReductor__p_threshold': 0.1, 'prep__VarianceExpressionReductor__v_threshold': 0.2}
model = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=15000,
    class_weight='balanced', l1_ratio = 0.1, C = 2, fit_intercept=True
)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('AnovaReductor', AnovaReductor()),
    ('MeanExpressionReductor', MeanExpressionReductor(3)),
    ('scaler', StandardScaler()),
    ('AgeBiasReductor', AgeBiasReductor()),
    ("rfe", RFE(estimator=model, n_features_to_select=1500, step=0.2, verbose = 1)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
y_proba = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

show_report(y_pred, y_test, ds, le)
print("Confusion Matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
plot_roc_curve(y_proba, y_test, "logistic regression")

print("f1 score: ", f1_score(y_test, y_pred, average="weighted"))




