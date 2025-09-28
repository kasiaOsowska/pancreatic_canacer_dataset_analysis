from Dataset import load_dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from utilz import *

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.X = ds.X[ARL2]
print(ds.X)

X_train, y_train, X_test, y_test = ds.training_split(test_size=0.2, random_state=42)
# combine healthy and disease into one class
y_train = y_train.replace({DISEASE: HEALTHY})
y_test = y_test.replace({DISEASE: HEALTHY})

X_train = np.array(X_train.values).reshape(-1, 1)
X_test = np.array(X_test.values).reshape(-1, 1)

print("------------")
print(X_train.shape, y_train.shape)

le = LabelEncoder()
y_train_encoded = pd.Series(le.fit_transform(y_train), index=y_train)
y_test_encoded = pd.Series(le.transform(y_test), index=y_test)
y_train_encoded = y_train_encoded.values
y_test_encoded = y_test_encoded.values

model = LogisticRegression()
model.fit(X_train, y_train_encoded)

y_pred = model.predict(X_test)

classification_report = classification_report(y_test_encoded, y_pred)
print(classification_report)

plt.figure()
plt.scatter(X_train, y_train, label="Train points", alpha=0.7)

x_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, label="Fitted line")

plt.xlabel("CDKN2A expression")
plt.ylabel("Label (encoded 0/1)")
plt.legend()
plt.title("Linear regression on binary label (for visualization)")
plt.show()