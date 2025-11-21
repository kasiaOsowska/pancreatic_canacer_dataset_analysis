import utilz
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.model_selection import train_test_split
from utilz import *
import matplotlib.pyplot as plt
from Dataset import load_dataset



meta_path = r"../data/samples_pancreatic_filtered.xlsx"
data_path = r"../data/counts_pancreatic_filtered.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})
X_train, X_test, y_train, y_test = train_test_split(ds.X, ds.y, test_size=0.5,
                                                    random_state=42, stratify=ds.y)

print("X train, y train shapes:")
print(X_train.shape, y_train.shape)
print("X test, y test shapes:")
print(X_test.shape, y_test.shape)

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

bst = XGBClassifier(n_estimators=220, max_depth=3, learning_rate=0.06, objective='binary:logistic',
                    colsample_bytree = 0.8, reg_lambda = 3.0, gamma = 0, min_child_weight = 1,
                    subsample = 0.9)
scaler = StandardScaler()
pipeline = Pipeline([('bst', bst)])

y_pred = cross_val_predict(pipeline, ds.X, y_encoded, cv=LeaveOneOut(), n_jobs=-1)

cm = confusion_matrix(y_encoded, y_pred, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Macierz pomyłek (Leave-One-Out)")
plt.show()
print("Macierz pomyłek:\n", cm)
print("\nRaport klasyfikacji:\n", classification_report(y_encoded, y_pred, target_names=le.classes_))

utilz.show_report(y_pred, y_encoded, ds, le)

