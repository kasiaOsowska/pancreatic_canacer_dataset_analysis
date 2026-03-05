from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utilz.preprocessing_utilz import *
from utilz.helpers import *
from utilz.Dataset import load_dataset


meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic_filtered_sex.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

# Prediction of age is only meaningful for healthy samples
healthy_idx = ds.y[ds.y == HEALTHY].index
VALID_SEX = {'F', 'M'}
y = ds.sex.loc[healthy_idx].dropna().astype(str)
y = y[y.isin(VALID_SEX)]
idx = y.index
X = ds.X.loc[idx]
X_train, X_test, X_valid, y_train, y_test, y_valid = ds.get_train_test_valid_split(y=y_encoded, test_size=0.2, valid_size=0.2)


print("X train, y train shapes:")
print(X_train.shape, y_train.shape)
print("X test, y test shapes:")
print(X_test.shape, y_test.shape)


le = LabelEncoder()
y_train_encoded = pd.Series(le.fit_transform(y_train), index=y_train.index)
y_test_encoded = pd.Series(le.transform(y_test), index=y_test.index)

"""
model = LogisticRegression(
    penalty='l1', solver='liblinear', max_iter=1500,
    class_weight='balanced', C=0.2, fit_intercept=True
)
"""

model = LogisticRegression(
    solver='saga', max_iter=1500,
    class_weight='balanced', l1_ratio = 0.1, C = 2, fit_intercept=True
)


scaler = StandardScaler()
pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('scaler', StandardScaler()),
    ('model', model)
])

pipeline.fit(X_train, y_train_encoded)
y_pred = pipeline.predict(X_test)
print("y test encoded:")
print(y_test_encoded)
print("y_pred")
show_report(y_pred, y_test_encoded, ds, le)

print(confusion_matrix(y_test_encoded, y_pred, labels=[0, 1]))
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))


coef = pipeline.named_steps["model"].coef_.ravel()
n_zero = (coef == 0).sum()
n_total = coef.size

print(f"Zerowe współczynniki: {n_zero}/{n_total} ({n_zero/n_total:.2%})")