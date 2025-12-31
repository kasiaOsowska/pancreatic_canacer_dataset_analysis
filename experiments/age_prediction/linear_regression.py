from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression
import shap

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

y = ds.age
y = y.dropna().astype(int)
X = ds.X.loc[y.index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=42)

model = LinearRegression(fit_intercept=True)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Mean Absolute Error:", np.mean(np.abs(y_test - y_pred)))
print("Mean Squared Error:", np.mean((y_test - y_pred) ** 2))
print("R^2 Score:", model.score(X_test, y_test))
