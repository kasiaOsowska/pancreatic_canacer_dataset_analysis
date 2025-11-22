from Dataset import load_dataset
from utilz import *

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from preprocessing_utilz import *



meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic_filtered.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

preprocessing_pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('VarianceExpressionReductor', VarianceExpressionReductor(0.1)),
    ('MeanExpressionReductor', MeanExpressionReductor(4)),
    ('PValueReductor', PValueReductor(0.005)),
    ('MinValueAdjustment', MinValueAdjustment("subtract")),
    ('scaler', StandardScaler())
])


X = preprocessing_pipeline.fit_transform(ds.X, y_encoded)

bst = XGBClassifier(objective='binary:logistic')

param_grid = {
    'bst__colsample_bytree': [0.7, 0.8],
    'bst__reg_lambda': [2.0, 3.0],
    'bst__gamma': [0, 1],
    'bst__min_child_weight': [1, 2],
    'bst__n_estimators': [200, 220],
    'bst__max_depth': [3, 4],
    'bst__learning_rate': [0.04, 0.05, 0.055, 0.06],
    'bst__subsample': [0.8, 0.9]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
print(cv)

grid = GridSearchCV(
    estimator=bst,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid.fit(X, y_encoded)
print("Najlepsze parametry:", grid.best_params_)
print("Najlepszy wynik CV (f1):", grid.best_score_)
