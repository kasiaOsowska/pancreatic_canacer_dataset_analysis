import utilz
from Dataset import load_dataset

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from utilz import *


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
y_train_encoded = pd.Series(le.fit_transform(y_train), index=y_train.index)
y_test_encoded = pd.Series(le.transform(y_test), index=y_test.index)

model = LogisticRegression(
    penalty='l2', solver='lbfgs', max_iter=500,
    class_weight='balanced',
)

scaler = StandardScaler()
pca = PCA(n_components=1, svd_solver='full')
pipeline = Pipeline([('scaler', scaler), ('model', model)])

y_train_pred = pipeline.fit(X_train, y_train_encoded)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)
print(y_proba)

utilz.show_report(y_pred, y_test_encoded, ds, le)

cm = confusion_matrix(y_test_encoded, y_pred, labels=le.transform(le.classes_))
cm_df = pd.DataFrame(
    cm,
    index=[f"Prawdziwe: {cls}" for cls in le.classes_],
    columns=[f"Przewidziane: {cls}" for cls in le.classes_]
)

print("\nMacierz pomyłek:")
print(cm_df)
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

