from collections import Counter
from scipy.stats import randint, uniform

from utilz.Dataset import load_dataset
from utilz.helpers import *
from utilz.preprocessing_utilz import *
from utilz.constans import DISEASE, HEALTHY

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)
class_counts = Counter(y_encoded)
scale_pos_weight = class_counts[0] / class_counts[1]

pipe = Pipeline([
    ('constant',  ConstantExpressionReductor()),
    ('high_disp', HighDispersionReductor()),
    ('mean_expr', MeanExpressionReductor(3)),
    ('age_bias',  CovariatesBiasReductor(covariate=ds.age)),
    ('sex_bias',  CovariatesBiasReductor(covariate=ds.sex)),
    ('anova',     AnovaReductor()),
    ('scaler',    StandardScaler()),
    ('clf',       XGBClassifier(scale_pos_weight=scale_pos_weight,
                                eval_metric='logloss', random_state=42, n_jobs=1)),
])

param_dist = {
    'clf__n_estimators':     randint(100, 500),
    'clf__max_depth':        randint(2, 7),
    'clf__learning_rate':    uniform(0.01, 0.1),
    'clf__gamma':            uniform(0, 2),
    'clf__reg_lambda':       uniform(1.0, 5.0),
    'clf__reg_alpha':        uniform(0.0, 2.0),
    'clf__colsample_bytree': uniform(0.5, 0.5),
    'clf__subsample':        uniform(0.5, 0.5),
    'clf__min_child_weight': randint(1, 10),
}

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=100,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True,
)

search.fit(ds.X, y_encoded)
print("Najlepsze parametry:", search.best_params_)
print("Najlepszy wynik CV (AUC):", search.best_score_)

#Najlepsze parametry: {'clf__colsample_bytree': np.float64(0.9056020883680015), 'clf__gamma': np.float64(1.6412789514879107), 'clf__learning_rate': np.float64(0.07259396701015727), 'clf__max_depth': 2, 'clf__min_child_weight': 1, 'clf__n_estimators': 267, 'clf__reg_alpha': np.float64(1.8107012839121275), 'clf__reg_lambda': np.float64(1.9789556739464822), 'clf__subsample': np.float64(0.5346806504375827)}
#Najlepszy wynik CV (AUC): 0.8602318840579709