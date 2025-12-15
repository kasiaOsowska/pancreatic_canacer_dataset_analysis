from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import utilz
from Dataset import load_dataset

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression

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

rus = RandomUnderSampler(random_state=42)
X_test, y_test = rus.fit_resample(X_test, y_test)

bst = XGBClassifier(scale_pos_weight=5.0, n_estimators=220, colsample_bytree= 0.8, gamma= 1, learning_rate=0.04,
max_depth= 4, min_child_weight= 2, reg_lambda= 3.0, subsample= 0.8)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('VarianceExpressionReductor', VarianceExpressionReductor(0.1)),
    ('MeanExpressionReductor', MeanExpressionReductor(4)),
    ('PValueReductor', PValueReductor(0.0005)),
    ('MinValueAdjustment', MinValueAdjustment("subtract")),
    ('scaler', StandardScaler()),
    ('model', bst)
])

#y_pred = cross_val_predict(pipeline, ds.X, y_encoded, cv=5, n_jobs=-1)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Macierz pomyłek (Leave-One-Out)")
plt.show()
utilz.show_report(y_pred, y_test, ds, le)
print("Macierz pomyłek:\n", cm)
print("\nRaport klasyfikacji:\n", classification_report(y_test, y_pred, target_names=le.classes_))

