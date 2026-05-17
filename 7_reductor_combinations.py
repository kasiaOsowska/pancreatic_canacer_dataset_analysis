"""
Wszystkie kombinacje reduktorow z utilz/preprocessing_utilz.

Idea:
  ConstantExpressionReductor jest zawsze (cheap, eliminuje 0-wariancyjne geny).
  Pozostale 5 reduktorow jest opcjonalne -> 2^5 = 32 podzbiory.
  Hiperparametry KAZDEGO reduktora ustalone (sensowne defaulty); zakladamy
  ze tuning hiperparametrow zrobiles w 6_reductor_benchmark.py.

  Kolejnosc w pipeline (kanoniczna, od taniego/ogolnego do drozszego/specyficznego):
    const -> MeanExpression -> Log2FC -> AnovaFdr -> MannWhitney -> WGV -> clf

Wyjscie:
  - reductor_combinations_results.csv  - 32 wiersze x 3 modele
  - reductor_combinations_auc.png      - scatter: liczba reduktorow vs CV AUC
  - reductor_combinations_top.png      - bar chart top-15 kombinacji
"""

import warnings
warnings.filterwarnings('ignore')

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
CV_FOLDS   = 5
N_JOBS     = -1

OUT_CSV       = "reductor_combinations_results.csv"
OUT_PNG_ALL   = "reductor_combinations_auc.png"
OUT_PNG_TOP   = "reductor_combinations_top.png"


# ---------------------------------------------------------------------------
# Reduktory: nazwa -> (klasa, kwargs)
# Kolejnosc w slowniku odpowiada KANONICZNEJ kolejnosci w pipeline.
# ---------------------------------------------------------------------------
def build_reductor_specs():
    return {
        'MeanExpr':    (MeanExpressionReductor,         {'percentile': 5}),
        'Log2FC':      (Log2FCReductor,                 {'min_abs_log2fc': np.log2(1.2)}),
        'AnovaFdr':    (AnovaFdrReductor,               {'alpha': 0.05}),
        'WGV':         (WithinGroupVarianceReductor,    {'alpha': 0.05}),
    }


def get_models(seed):
    return {
        'LogReg': LogisticRegression(
            penalty='l2', C=1.0, solver='saga',
            max_iter=20000, class_weight='balanced',
            random_state=seed,
        ),
    }


def build_pipeline(subset_names, reductor_specs, model):
    """Buduje pipeline w KOLEJNOSCI z reductor_specs (po podzbiorze)."""
    steps = []
    for name in reductor_specs:                       # iter w kanonicznej kolejnosci
        if name in subset_names:
            cls, kwargs = reductor_specs[name]
            steps.append((name, cls(**kwargs)))
    steps.append(('scaler', StandardScaler()))
    steps.append(('clf',    clone(model)))
    return Pipeline(steps)


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
    const = ConstantExpressionReductor()
    X_tr_raw = const.fit_transform(X_tr_raw, y_train)
    X_te_raw = const.transform(X_te_raw)
    X_va_raw = const.transform(X_va_raw)
    X_te_raw = pd.concat([X_te_raw, X_va_raw])
    y_test   = pd.concat([y_test, y_valid])
    print(f"Train: {len(X_tr_raw)}  cancer={int(y_train.sum())} ctrl={int((y_train==0).sum())}")
    print(f"Test:  {len(X_te_raw)}  cancer={int(y_test.sum())}  ctrl={int((y_test==0).sum())}")

    cv = StratifiedKFold(
        n_splits=min(CV_FOLDS, int(np.bincount(y_train.astype(int)).min())),
        shuffle=True, random_state=BASE_SEED,
    )

    reductor_specs = build_reductor_specs()
    reductor_names = list(reductor_specs.keys())

    # generuj wszystkie 2^5 = 32 podzbiory
    all_subsets = []
    for r in range(0, len(reductor_names) + 1):
        for combo in itertools.combinations(reductor_names, r):
            all_subsets.append(combo)
    print(f"\nLiczba kombinacji: {len(all_subsets)}  (2^{len(reductor_names)} podzbiorow)")

    models = get_models(BASE_SEED)
    rows = []
    total = len(all_subsets) * len(models)
    counter = 0
    for model_name, model in models.items():
        for subset in all_subsets:
            counter += 1
            label = '+'.join(subset) if subset else 'const_only'
            pipe = build_pipeline(subset, reductor_specs, model)

            scores = cross_val_score(
                pipe, X_tr_raw, y_train,
                scoring='roc_auc', cv=cv, n_jobs=N_JOBS,
            )
            cv_mean = float(scores.mean())
            cv_std  = float(scores.std(ddof=1))

            # refit + test AUC
            pipe.fit(X_tr_raw, y_train)
            test_auc = float(roc_auc_score(
                y_test, pipe.predict_proba(X_te_raw)[:, 1]
            ))

            rows.append({
                'model':       model_name,
                'n_reductors': len(subset),
                'combo':       label,
                'cv_auc':      cv_mean,
                'cv_auc_std':  cv_std,
                'test_auc':    test_auc,
            })
            print(f"[{counter:>3}/{total}]  {model_name:6s}  {label:60s}  "
                  f"CV={cv_mean:.4f}+-{cv_std:.4f}  Test={test_auc:.4f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] wyniki -> {OUT_CSV}")

    # === podsumowanie: best per model ===
    print("\n=== BEST PER MODEL ===")
    for model_name in models:
        sub = results_df[results_df['model'] == model_name]
        best = sub.loc[sub['cv_auc'].idxmax()]
        print(f"  {model_name:6s}: CV={best['cv_auc']:.4f}  Test={best['test_auc']:.4f}  "
              f"combo='{best['combo']}'")

    # === wykres 1: scatter (n_reductors vs CV AUC) ===
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'LogReg': '#6366f1', 'XGB': '#10b981', 'LGBM': '#f59e0b'}
    for model_name in models:
        sub = results_df[results_df['model'] == model_name]
        ax.scatter(sub['n_reductors'] + np.random.uniform(-0.1, 0.1, len(sub)),
                   sub['cv_auc'], label=model_name, alpha=0.7,
                   color=colors.get(model_name, 'gray'), s=40)
    ax.set_xlabel('Liczba reduktorow w pipeline (oprocz const)')
    ax.set_ylabel('CV AUC (train)')
    ax.set_title(f'Wszystkie {len(all_subsets)} kombinacje reduktorow')
    ax.set_xticks(range(0, len(reductor_names) + 1))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_ALL, dpi=140)
    plt.show()
    print(f"[OK] scatter -> {OUT_PNG_ALL}")

    # === wykres 2: top-15 kombinacji per model (slupki) ===
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 4*len(models)), sharex=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, models):
        sub = (results_df[results_df['model'] == model_name]
               .sort_values('cv_auc', ascending=False).head(15))
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub['cv_auc'], xerr=sub['cv_auc_std'],
                color=colors.get(model_name, 'gray'), alpha=0.85, capsize=3)
        for i, (cv_v, te_v) in enumerate(zip(sub['cv_auc'], sub['test_auc'])):
            ax.text(cv_v + 0.005, i, f'CV={cv_v:.3f}  T={te_v:.3f}',
                    va='center', fontsize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub['combo'].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.5, 1.05)
        ax.set_xlabel('CV AUC (± std)')
        ax.set_title(f'Top-15 kombinacji dla {model_name}')
        ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUT_PNG_TOP, dpi=140)
    plt.show()
    print(f"[OK] top-15 -> {OUT_PNG_TOP}")


if __name__ == '__main__':
    main()
