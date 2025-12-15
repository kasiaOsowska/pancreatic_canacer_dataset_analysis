from imblearn.over_sampling import RandomOverSampler

import utilz
from Dataset import load_dataset

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler


from preprocessing_utilz import *
from utilz import *


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

rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)



class_weights = {0: 1, 1: 2}
model = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=1500,
    class_weight=class_weights, l1_ratio = 0.8, C = 2
)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('VarianceExpressionReductor', VarianceExpressionReductor(0.1)),
    ('MeanExpressionReductor', MeanExpressionReductor(4)),
    ('PValueReductor', PValueReductor(0.005)),
    ('MinValueAdjustment', MinValueAdjustment("subtract")),
    ('scaler', StandardScaler()),
    #('PCA', PCA(n_components=20)),
    ('model', model)
])

#y_pred = cross_val_predict(pipeline, ds.X, y_encoded, cv=5, n_jobs=-1)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.show()
utilz.show_report(y_pred, y_test, ds, le)

print("Confusion Matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


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

import shap


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