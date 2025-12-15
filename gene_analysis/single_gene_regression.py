from sklearn.model_selection import train_test_split

from Dataset import load_dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from utilz import *

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.X = ds.X[MYL9]
print(ds.X)

ds.y =  ds.y.replace({DISEASE: HEALTHY})
le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y)
X_train, X_test, y_train, y_test = train_test_split(ds.X, y_encoded, test_size=0.2,
                                                    random_state=42, stratify=y_encoded)


X_train = np.array(X_train.values).reshape(-1, 1)
X_test = np.array(X_test.values).reshape(-1, 1)

print("------------")
print(X_train.shape, y_train.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

classification_report = classification_report(y_test, y_pred)
print(classification_report)
print("f1 score" + str(f1_score(y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion matrix:\n", cm)
plt.figure()
plt.scatter(X_train, y_train, label="Train points", alpha=0.7)

x_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, label="Fitted line")

plt.xlabel("BCAP31 expression")
plt.ylabel("Label (encoded 0/1)")
plt.legend()
plt.title("Logistic regression on binary label (for visualization)")
plt.show()