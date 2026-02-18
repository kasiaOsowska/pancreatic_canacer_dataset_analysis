from collections import Counter

from utilz.Dataset import load_dataset
from utilz.helpers import *

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from utilz.preprocessing_utilz import *
from utilz.constans import DISEASE, HEALTHY


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)
class_counts = Counter(y_encoded)
scale_pos_weight = class_counts[0] / class_counts[1]

preprocessing_pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('VarianceExpressionReductor', AnovaReductor(0.2)),
    ('MeanExpressionReductor', MeanExpressionReductor(3)),
    ('PValueReductor', PValueReductor(0.1)),
    ('MinValueAdjustment', MinValueAdjustment("subtract_all")),
    ('scaler', StandardScaler()),
])

full_pipeline = Pipeline([
    ('prep', preprocessing_pipeline),
    ('bst', XGBClassifier(scale_pos_weight = scale_pos_weight))
])

X = preprocessing_pipeline.fit_transform(ds.X, y_encoded)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

param_grid = {
    'bst__colsample_bytree': [0.7, 0.8],
    'bst__reg_lambda': [2.8, 3.0],
    'bst__gamma': [0.8, 1],
    'bst__min_child_weight': [2, 3],
    'bst__n_estimators': [220, 250],
    'bst__max_depth': [3, 4],
    'bst__learning_rate': [0.03, 0.04],
    'bst__subsample': [0.7, 0.8]
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

"""
Najlepsze parametry: {'bst__colsample_bytree': 0.8, 'bst__gamma': 1, 'bst__learning_rate': 0.04, 'bst__max_depth': 4, 
'bst__min_child_weight': 2, 'bst__n_estimators': 220, 'bst__reg_lambda': 3.0, 'bst__subsample': 0.8}
Najlepszy wynik CV (f1): 0.5339986839219126

"""
