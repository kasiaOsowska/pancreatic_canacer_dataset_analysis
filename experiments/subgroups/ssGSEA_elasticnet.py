import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY

meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds_pathways = ds.get_pathway_scores(
    library='Reactome_2022',
    cache_dir='./cache/reactome'
)

X_pathways = ds_pathways.X.values
y = ds_pathways.y.replace({DISEASE: HEALTHY})

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegressionCV(
        Cs=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        penalty='elasticnet',
        solver='saga',
        l1_ratios=[0.1, 0.5, 0.9],
        scoring='roc_auc',
        max_iter=2000,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_pathways, y)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
auc_scores = cross_val_score(model, X_pathways, y, cv=outer_cv, scoring='roc_auc')
print(f"AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

pathway_names = ds_pathways.X.columns.tolist()
coefs = model.named_steps['clf'].coef_[0]

results = (
    pd.DataFrame({'pathway': pathway_names, 'coefficient': coefs})
    .sort_values('coefficient', key=abs, ascending=False)
)

selected = results[results['coefficient'] != 0]
print(f"Wybrano {len(selected)} ścieżek z {len(pathway_names)}")
print(selected.head(20))