from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from utilz.Dataset import load_dataset
from utilz.constans import *
from utilz.preprocessing_utilz import *

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)
sex_numeric = ds.sex.map({"F": 0, "M": 1})

X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(
        ds.X, y_encoded, test_size=0.25, valid_size=0.25
    )
)
train_idx = X_train.index

# ── Step 1: preprocessing POZA SFS ──────────────────────────────────────────
# CovariatesBiasReductor dostaje tylko dane treningowe — brak leakage
preproc = Pipeline([
    ('constant',  ConstantExpressionReductor()),
    ('anova',     AnovaReductor(percentile=70)),
    ('mean_expr', MeanExpressionReductor(percentile=10)),
    ('age_bias',  CovariatesBiasReductor(covariate=ds.age.loc[train_idx])),
    ('sex_bias',  CovariatesBiasReductor(covariate=sex_numeric.loc[train_idx])),
])

# ogranicz do genów spoza [SCN1B, MAGOHB] i przefiltruj
X_candidates = ds.X[[g for g in ds.X.columns if g not in [SCN1B, MAGOHB]]]
X_train_pre = preproc.fit_transform(X_candidates.loc[train_idx], y_train)

# odtwórz nazwy genów które przeżyły preprocessing
try:
    survived_genes = preproc.get_feature_names_out().tolist()
except AttributeError:
    # fallback jeśli custom reducery nie implementują get_feature_names_out
    survived_mask = preproc.named_steps['anova'].get_support()  # dostosuj do API
    survived_genes = X_candidates.columns[survived_mask].tolist()

X_train_pre = pd.DataFrame(X_train_pre, columns=survived_genes, index=train_idx)

print(f"Genów po preprocessing: {X_train_pre.shape[1]}")

# ── Step 2: SFS tylko z scalerem i modelem ───────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

sfs_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(
        max_iter=1500, class_weight='balanced', fit_intercept=True
    )),
])

sfs = SequentialFeatureSelector(
    estimator=sfs_estimator,
    n_features_to_select=1,       # 5 nowych + 2 startowe = 7 łącznie
    direction='forward',
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
)

sfs.fit(X_train_pre, y_train)

# ── Step 3: wyniki ───────────────────────────────────────────────────────────
new_genes     = X_train_pre.columns[sfs.get_support()].tolist()
final_features = [SCN1B, MAGOHB] + new_genes

print("Nowe geny wybrane przez SFS:", new_genes)
print("Finalne cechy:", final_features)

# ['ENSG00000105711', 'ENSG00000111196', 'ENSG00000119632', 'ENSG00000131019', 'ENSG00000132294', 'ENSG00000151490', 'ENSG00000260661']