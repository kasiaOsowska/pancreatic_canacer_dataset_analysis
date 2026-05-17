"""
Benchmark metod redukcji wymiarow z utilz/preprocessing_utilz.

Schemat:
  Pipeline = [ConstantExpressionReductor, <reductor switch>, StandardScaler, clf]
  GridSearchCV iteruje po WSZYSTKICH metodach redukcji i ich hiperparametrach;
  modele bazowe (LogReg, XGB, LGBM) bez tuningu - chcemy zmierzyc *wplyw
  redukcji* na CV AUC, a nie tuningu hiperparametrow modelu.

Wyjscie:
  - reductor_benchmark_results.csv  - pelna siatka (model x reductor x params)
  - reductor_benchmark_auc.png      - bar chart: best CV AUC per (model, reductor)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY
from utilz.preprocessing_utilz import (
    ConstantExpressionReductor, AnovaFdrReductor, Log2FCReductor,
    MannWhitneyReductor, MeanExpressionReductor, WithinGroupVarianceReductor,
)

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
meta_path  = r"../data/samples_pancreatic.xlsx"
data_path  = r"../data/counts_pancreatic.csv"

TEST_SIZE  = 0.2
VALID_SIZE = 0.2
BASE_SEED  = 2137
CV_FOLDS   = 10
N_JOBS     = -1

OUT_CSV    = "reductor_benchmark_results.csv"
OUT_PNG    = "reductor_benchmark_auc.png"


# ---------------------------------------------------------------------------
# Modele bazowe - sensowne defaulty, BEZ tuningu (porownujemy reduktory)
# ---------------------------------------------------------------------------
def get_models(seed):
    return {
        'LogReg': LogisticRegression(
            penalty='l2', C=1.0, solver='saga',
            max_iter=20000, class_weight='balanced',
            random_state=seed,
        ),
        'XGB': XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            tree_method='hist',
            n_estimators=300, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=1, verbosity=0,
        ),
        'LGBM': LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            num_leaves=31, class_weight='balanced',
            random_state=seed, n_jobs=1, verbose=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Siatka reduktorow - kazdy slownik to "switch case" dla GridSearchCV.
# sklearn robi kartezjan w obrebie slownika, a sume miedzy slownikami.
# ---------------------------------------------------------------------------
def get_reductor_grid():
    return [
        # Brak redukcji (baseline)
        {'reductor': ['passthrough']},

        {
            'reductor': [AnovaFdrReductor()],
            'reductor__alpha': [0.01, 0.05, 0.1],
        },
        {
            'reductor': [Log2FCReductor()],
            'reductor__min_abs_log2fc': [np.log2(1.2), np.log2(1.4)],
        },
        {
            'reductor': [MannWhitneyReductor()],
            'reductor__alpha': [0.01, 0.05, 0.1],
        },
        {
            'reductor': [MeanExpressionReductor()],
            'reductor__percentile': [5, 10, 25],
        },
        {
            'reductor': [WithinGroupVarianceReductor()],
            'reductor__alpha': [0.01, 0.05, 0.1],
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # === split (taki sam jak inne skrypty PLA2Sig) ===
    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})
    y_enc = pd.Series(LabelEncoder().fit_transform(ds.y), index=ds.y.index)

    X_tr_raw, X_te_raw, X_va_raw, y_train, y_test, y_valid = ds.get_train_test_valid_split(
        ds.X, y_enc, test_size=TEST_SIZE, valid_size=VALID_SIZE,
        random_state=BASE_SEED,
    )
    # walidacyjny niepotrzebny -> doklejamy do test setu (split deterministyczny)
    X_te_raw = pd.concat([X_te_raw, X_va_raw])
    y_test   = pd.concat([y_test, y_valid])
    print(f"Train: {len(X_tr_raw)}  cancer={int(y_train.sum())} ctrl={int((y_train==0).sum())}")
    print(f"Test:  {len(X_te_raw)}  cancer={int(y_test.sum())}  ctrl={int((y_test==0).sum())}")

    cv = StratifiedKFold(
        n_splits=min(CV_FOLDS, int(np.bincount(y_train.astype(int)).min())),
        shuffle=True, random_state=BASE_SEED,
    )

    all_rows   = []
    best_rows  = []
    for model_name, model in get_models(BASE_SEED).items():
        print(f"\n=========================================")
        print(f"  {model_name}")
        print(f"=========================================")
        pipe = Pipeline([
            ('const',    ConstantExpressionReductor()),
            ('reductor', 'passthrough'),
            ('scaler',   StandardScaler()),
            ('clf',      model),
        ])
        gs = GridSearchCV(
            pipe, param_grid=get_reductor_grid(),
            scoring='roc_auc', cv=cv, n_jobs=N_JOBS, refit=True,
            return_train_score=False, verbose=1,
        ).fit(X_tr_raw, y_train)

        cv_results = pd.DataFrame(gs.cv_results_)
        for _, r in cv_results.iterrows():
            params = r['params']
            reductor_obj = params.get('reductor', 'passthrough')
            reductor_name = (type(reductor_obj).__name__
                             if reductor_obj != 'passthrough' else 'NoReduction')
            reductor_params = {k.replace('reductor__', ''): v
                               for k, v in params.items() if k.startswith('reductor__')}
            all_rows.append({
                'model':           model_name,
                'reductor':        reductor_name,
                'reductor_params': str(reductor_params),
                'cv_auc':          float(r['mean_test_score']),
                'cv_auc_std':      float(r['std_test_score']),
            })

        # holdout test AUC dla best_estimator_
        proba_te = gs.best_estimator_.predict_proba(X_te_raw)[:, 1]
        test_auc = float(roc_auc_score(y_test, proba_te))

        best_params = gs.best_params_
        best_reductor = best_params.get('reductor', 'passthrough')
        best_reductor_name = (type(best_reductor).__name__
                              if best_reductor != 'passthrough' else 'NoReduction')
        best_reductor_params = {k.replace('reductor__', ''): v
                                for k, v in best_params.items()
                                if k.startswith('reductor__')}
        best_rows.append({
            'model':           model_name,
            'reductor':        best_reductor_name,
            'reductor_params': str(best_reductor_params),
            'cv_auc':          float(gs.best_score_),
            'test_auc':        test_auc,
        })
        print(f"\nBest dla {model_name}: {best_reductor_name} {best_reductor_params}")
        print(f"  CV AUC   = {gs.best_score_:.4f}")
        print(f"  Test AUC = {test_auc:.4f}")

    # === zapis ===
    results_df = pd.DataFrame(all_rows).sort_values(
        ['model', 'cv_auc'], ascending=[True, False]
    ).reset_index(drop=True)
    best_df = pd.DataFrame(best_rows)

    results_df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] pelne wyniki -> {OUT_CSV}")
    print("\n=== BEST PER MODEL ===")
    print(best_df.to_string(index=False))

    # === wykres: best CV AUC per (model, reductor) ===
    pivot = (results_df.groupby(['model', 'reductor'])['cv_auc']
             .max().unstack('model'))
    # ustal kolejnosc reduktorow
    order = ['NoReduction', 'AnovaFdrReductor', 'Log2FCReductor',
             'MannWhitneyReductor', 'MeanExpressionReductor',
             'WithinGroupVarianceReductor']
    pivot = pivot.reindex([r for r in order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind='bar', ax=ax, width=0.78)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=7, padding=2)
    ax.set_ylabel('Best CV AUC (po hiperparametrach reduktora)')
    ax.set_xlabel('Metoda redukcji')
    ax.set_title('Wplyw metody redukcji na CV AUC (modele bazowe, bez tuningu)')
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(title='Model', loc='lower right')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)
    plt.show()
    print(f"[OK] wykres -> {OUT_PNG}")


if __name__ == '__main__':
    main()
