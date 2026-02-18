from collections import Counter

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

X_train, X_test, y_train, y_test = train_test_split(ds.X, y_encoded, test_size=0.5,
                                                    random_state=42, stratify=y_encoded)

class_counts = Counter(y_encoded)
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"Stosunek klas: {scale_pos_weight:.2f}")
bst = XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=220, colsample_bytree= 0.8, gamma= 1,
                    learning_rate=0.04, max_depth= 4, min_child_weight= 2, reg_lambda= 3.0, subsample= 0.8)

print("original X shape: ", X_train.shape)
pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('AnovaReductor', AnovaReductor()),
    ('MeanExpressionReductor', MeanExpressionReductor(3)),
    ('scaler', StandardScaler()),
    ('model', bst)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Macierz pomyłek (Leave-One-Out)")
plt.show()
show_report(y_pred, y_test, ds, le)
print("Macierz pomyłek:\n", cm)
print("\nRaport klasyfikacji:\n", classification_report(y_test, y_pred, target_names=le.classes_))

