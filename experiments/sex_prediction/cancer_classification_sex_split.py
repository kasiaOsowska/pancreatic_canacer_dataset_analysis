from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utilz.constans import DISEASE, HEALTHY
from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *

meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

ds.y_female = ds.y_female.replace({DISEASE: HEALTHY})
ds.y_male = ds.y_male.replace({DISEASE: HEALTHY})
X_female = ds.X_female
y_female = ds.y_female
X_male   = ds.X_male
y_male   = ds.y_male


le = LabelEncoder()
y_female_encoded = pd.Series(le.fit_transform(y_female), index=y_female.index)
y_male_encoded = pd.Series(le.transform(y_male), index=y_male.index)

rus = RandomUnderSampler(random_state=42)
X_female, y_female_encoded = rus.fit_resample(X_female, y_female_encoded)

rus = RandomUnderSampler(random_state=42)
X_male, y_male_encoded = rus.fit_resample(X_male, y_male_encoded)

# FEMALE CLASSIFICATION -----------------------
model_female = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=1500,
    class_weight='balanced', l1_ratio = 0.8, C = 2
)
pipeline_female = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('AnovaReductor', AnovaReductor()),
    ('MeanExpressionReductor', MeanExpressionReductor(4)),
    ('scaler', StandardScaler()),
    ('model_female', model_female)
])
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female, y_female_encoded, test_size=0.5,
                                                    random_state=42, stratify=y_female_encoded)

pipeline_female.fit(X_train_female, y_train_female)
y_pred_female = pipeline_female.predict(X_test_female)
cm = confusion_matrix(y_test_female, y_pred_female, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion matrix for female")
plt.show()
print("Confusion matrix for female\n", cm)
print("Classification report:\n", classification_report(y_test_female,
                                                        y_pred_female, target_names=le.classes_))


# MALE CLASSIFICATION -----------------------
model_male = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=1500,
    class_weight='balanced', l1_ratio = 0.8, C = 2
)

pipeline_male = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('AnovaReductor', AnovaReductor()),
    ('MeanExpressionReductor', MeanExpressionReductor(4)),
    ('scaler', StandardScaler()),
    ('model_female', model_female)
])

X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male, y_male_encoded, test_size=0.5,
                                                    random_state=42, stratify=y_male_encoded)

pipeline_male.fit(X_train_male, y_train_male)
y_pred_male = pipeline_male.predict(X_test_male)

cm = confusion_matrix(y_test_male, y_pred_male, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion matrix for male")
plt.show()
print("Confusion matrix for male:\n", cm)
print("Classification report:\n", classification_report(y_test_male, y_pred_male, target_names=le.classes_))


# Aggregate results
y_true_agg = pd.concat([y_test_female, y_test_male])
y_pred_agg = np.concatenate([y_pred_female, y_pred_male])
cm = confusion_matrix(y_true_agg, y_pred_agg,  labels=range(len(le.classes_)))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion matrix for both")
plt.show()
print("Confusion matrix for both:\n", cm)
print("Classification report:\n", classification_report(y_true_agg, y_pred_agg, target_names=le.classes_))
