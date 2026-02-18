from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import shap

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


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

print("original X shape: ", X_train.shape)
preprocessing_pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('VarianceExpressionReductor', AnovaReductor(0.1)),
    ('MeanExpressionReductor', MeanExpressionReductor(4)),
    ('PValueReductor', PValueReductor(0.01)),
    ('scaler', StandardScaler())
])

model = LogisticRegression(
    penalty='elasticnet', solver='saga', max_iter=15000,
    class_weight='balanced', l1_ratio=0.2
)

full_pipeline = Pipeline([
    ('prep', preprocessing_pipeline),
    ('model', model)
])

X = preprocessing_pipeline.fit_transform(ds.X, y_encoded)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

param_grid = {
    'model__l1_ratio': [0.2, 0.1, 0.05],
    'prep__MeanExpressionReductor__mean_threshold': [2, 3, 4, 5],
}


grid = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid.fit(ds.X, y_encoded)
print("Najlepsze parametry:", grid.best_params_)
print("Najlepszy wynik CV (f1):", grid.best_score_)

#Najlepsze parametry: {'model__l1_ratio': 0.1, 'prep__MeanExpressionReductor__mean_threshold': 3, 'prep__PValueReductor__p_threshold': 0.1, 'prep__VarianceExpressionReductor__v_threshold': 0.2}
#Najlepszy wynik CV (f1): 0.6571428571428571
