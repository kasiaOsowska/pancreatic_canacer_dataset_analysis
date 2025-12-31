from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from utilz.constans import *
from utilz.Dataset import load_dataset

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
selected_genes = [MYL9, BCAP31, ARL2, CFL1]
ds.X = ds.X[selected_genes]
print(ds.X)

ds.y =  ds.y.replace({DISEASE: HEALTHY})
le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)
X_train, X_test, y_train, y_test = train_test_split(ds.X, y_encoded, test_size=0.2,
                                                    random_state=42, stratify=y_encoded)


X_train = X_train.values
X_test = X_test.values

print("X train shape:", X_train.shape)
print("y_train shape", y_train.shape)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

classification_report = classification_report(y_test, y_pred)
print(classification_report)
print("f1 score" + str(f1_score(y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion matrix:\n", cm)
