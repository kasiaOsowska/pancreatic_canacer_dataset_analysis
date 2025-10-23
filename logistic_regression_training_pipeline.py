import utilz
from Dataset import load_dataset

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from utilz import *


meta_path = r"../data/samples_pancreatic_filtered.xlsx"
data_path = r"../data/counts_pancreatic_filtered.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

model = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=1500,
    class_weight='balanced', l1_ratio = 0.8,
)

scaler = StandardScaler()
pipeline = Pipeline([('scaler', scaler), ('model', model)])

y_pred = cross_val_predict(pipeline, ds.X, y_encoded, cv=5, n_jobs=-1)
cm = confusion_matrix(y_encoded, y_pred, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Macierz pomyłek (Leave-One-Out)")
plt.show()
utilz.show_report(y_pred, y_encoded, ds, le)

print("Macierz pomyłek:\n", cm)
print("\nRaport klasyfikacji:\n", classification_report(y_encoded, y_pred, target_names=le.classes_))


ds.y_female = ds.y_female.replace({DISEASE: HEALTHY})
ds.y_male = ds.y_male.replace({DISEASE: HEALTHY})
X_female = ds.X_female
y_female = ds.y_female
X_male   = ds.X_male
y_male   = ds.y_male

le_female = LabelEncoder()
le_male = LabelEncoder()
y_female_encoded = pd.Series(le_female.fit_transform(y_female), index=y_female.index)
y_male_encoded = pd.Series(le_male.fit_transform(y_male), index=y_male.index)

model_female = LogisticRegression(
    penalty='l2', solver='lbfgs', max_iter=1000,
    class_weight='balanced',
)

scaler_female = StandardScaler()
pipeline_female = Pipeline([('scaler', scaler_female), ('model', model_female)])

y_pred_female = cross_val_predict(pipeline_female, X_female, y_female_encoded, cv=LeaveOneOut(), n_jobs=-1)
cm = confusion_matrix(y_female_encoded, y_pred_female, labels=range(len(le_female.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_female.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Macierz pomyłek (Leave-One-Out) female")
plt.show()
print("Macierz pomyłek:\n", cm)
print("\nRaport klasyfikacji:\n", classification_report(y_female_encoded,
                                                        y_pred_female, target_names=le_female.classes_))


model_male = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=1000,
    class_weight='balanced', l1_ratio = 0.5,
)

scaler_male = StandardScaler()
pipeline_male = Pipeline([('scaler', scaler_male), ('model', model_male)])
y_pred_male = cross_val_predict(pipeline_male, X_male, y_male_encoded, cv=LeaveOneOut(), n_jobs=-1)
cm = confusion_matrix(y_male_encoded, y_pred_male, labels=range(len(le_male.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_male.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Macierz pomyłek (Leave-One-Out) male")
plt.show()
print("Macierz pomyłek:\n", cm)
print("\nRaport klasyfikacji:\n", classification_report(y_male_encoded, y_pred_male, target_names=le_male.classes_))
