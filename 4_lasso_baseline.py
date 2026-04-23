"""
4. Simple LASSO baseline (bez stability selection, bez RFE)

Cel: dac twardy punkt odniesienia dla pipeline'u stability + RFE.
Jesli ten baseline osiaga podobne AUC jak pelny pipeline, oznacza to ze
bardziej skomplikowane metody nie dodaja wartosci (albo dataset ma taki ceiling).

Schemat:
0. Ten sam holdout split co w 3_random_resampling_gene_stability.py (ten sam BASE_SEED)
1. Minimalny preprocessing na TRAIN: drop staly + scaler
2. LogisticRegressionCV z penalty='l1' (LASSO) -- C dobierane przez CV na TRAIN
3. Ewaluacja na czystym holdout test
4. Wypisanie top-N niezerowych wspolczynnikow
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY
from utilz.preprocessing_utilz import ConstantExpressionReductor

# ── paths ────────────────────────────────────────────────────────
meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

# ── hyperparameters (spojne z 3_random_resampling_gene_stability.py) ─
HOLDOUT_TEST_SIZE = 0.2
BASE_SEED         = 2137
CV_FOLDS          = 3          # CV na train do wyboru C
N_CS              = 10         # liczba wartosci C przetestowanych przez LogisticRegressionCV
TOP_N_COEFFS      = 30         # ile top niezerowych wspolczynnikow wypisac
K_SWEEP           = [2, 5, 10, 12, 15, 20]  # rozmiary podzbiorow do fair-compare
TARGET_K          = 12         # k odpowiadajacy TOP_K_FINAL z pipeline'u stability+RFE


def build_minimal_pipeline():
    """Minimalny preprocessing -- bez selekcji genow, bez residualizacji kowariat."""
    return Pipeline([
        ('ConstantExpressionReductor', ConstantExpressionReductor()),
        ('scaler',                     StandardScaler()),
    ])


def main():
    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

    print("Class mapping:")
    for cls in le.classes_:
        print(f"  {cls} -> {le.transform([cls])[0]}")
    print("\nClass distribution:")
    print(y_encoded.value_counts().sort_index())

    # ── 0. Holdout split (ten sam seed co pipeline ze stability) ─
    X_train_raw, X_test_raw, y_train, y_test = ds.get_train_test_valid_split(
        ds.X, y_encoded, test_size=HOLDOUT_TEST_SIZE, valid_size=0,
        random_state=BASE_SEED, return_valid=False,
    )
    print(f"\nHoldout split:")
    print(f"  Train: {len(X_train_raw)}  (cancer={int(y_train.sum())}, "
          f"ctrl={int((y_train==0).sum())})")
    print(f"  Test:  {len(X_test_raw)}   (cancer={int(y_test.sum())},  "
          f"ctrl={int((y_test==0).sum())})  <- NIE dotykany podczas tuningu")

    # ── 1. Minimalny preprocessing (fit TYLKO na train) ─────────
    pipe = build_minimal_pipeline()
    X_tr = pipe.fit_transform(X_train_raw, y_train)
    X_te = pipe.transform(X_test_raw)

    # feature names po ConstantExpressionReductor
    const_step = pipe.named_steps['ConstantExpressionReductor']
    if hasattr(const_step, 'selected_genes_'):
        feat_names = list(const_step.selected_genes_)
    else:
        feat_names = list(X_train_raw.columns)
    print(f"\nLiczba genow po dropie stalych: {X_tr.shape[1]}")

    # ── 2. LASSO z CV na train ──────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=BASE_SEED)
    lasso_cv = LogisticRegressionCV(
        Cs=N_CS,
        cv=cv,
        penalty='l1',
        solver='saga',
        scoring='roc_auc',
        class_weight='balanced',
        max_iter=20000,
        n_jobs=-1,
        random_state=BASE_SEED,
        refit=True,
    )
    print(f"\nTrening LASSO (LogRegCV, l1, {CV_FOLDS}-fold CV, {N_CS} wartosci C)...")
    lasso_cv.fit(X_tr, y_train)

    best_C = lasso_cv.C_[0]
    mean_cv_auc = lasso_cv.scores_[1].mean(axis=0)     # scores_ klucz = klasa 1
    best_cv_auc = mean_cv_auc.max()
    print(f"  Best C         = {best_C:.5g}")
    print(f"  Best CV AUC    = {best_cv_auc:.4f}  (mean over {CV_FOLDS} folds)")

    coefs = lasso_cv.coef_.ravel()
    n_nonzero = int(np.sum(coefs != 0))
    print(f"  Niezerowe wsp. = {n_nonzero} / {len(coefs)}")

    # ── 3. Ewaluacja na holdout test ────────────────────────────
    proba_te = lasso_cv.predict_proba(X_te)[:, 1]
    holdout_auc = roc_auc_score(y_test, proba_te)
    print(f"\n{'='*60}")
    print(f"HOLDOUT TEST AUC (LASSO, wszystkie geny) = {holdout_auc:.4f}")
    print('='*60)

    # ── dodatkowy porownawczy model: plain LogReg L2 bez tuningu ─
    logreg_plain = LogisticRegression(
        solver='saga', max_iter=20000,
        class_weight='balanced', random_state=BASE_SEED,
    )
    logreg_plain.fit(X_tr, y_train)
    plain_auc = roc_auc_score(y_test, logreg_plain.predict_proba(X_te)[:, 1])
    print(f"  (ref) Plain LogReg L2 (default C=1)  AUC = {plain_auc:.4f}")

    # ── 4. Top niezerowe wspolczynniki ──────────────────────────
    if len(feat_names) != len(coefs):
        # na wszelki wypadek gdyby ConstantExpressionReductor nie wystawial selected_genes_
        feat_names = [f"f{i}" for i in range(len(coefs))]

    coef_df = (
        pd.DataFrame({'gene': feat_names, 'coef': coefs})
        .assign(abs_coef=lambda df: df['coef'].abs())
        .query('coef != 0')
        .sort_values('abs_coef', ascending=False)
        .reset_index(drop=True)
    )
    coef_df.index += 1
    coef_df.index.name = 'rank'
    coef_df.to_csv('lasso_baseline_coefficients.csv', index=True)

    print(f"\nTop-{min(TOP_N_COEFFS, len(coef_df))} niezerowe wspolczynniki "
          f"(|coef| malejaco):")
    print(coef_df.head(TOP_N_COEFFS)[['gene', 'coef']].to_string())

    # ── 5. Fair-compare: baseline przy ograniczonej liczbie genow ─
    # LASSO wybiera ~n_nonzero cech -- zeby porownac z pipeline'em 12-genowym
    # refitujemy prosty LogReg L2 na top-k cechach wg |coef| LASSO, dla roznych k.
    print(f"\n{'='*60}")
    print("FAIR-COMPARE: top-k genow wg |coef| LASSO -> refit LogReg L2")
    print('='*60)

    feat_to_idx = {name: i for i, name in enumerate(feat_names)}
    sweep_rows = []
    for k in K_SWEEP:
        if k > len(coef_df):
            continue
        top_k_genes = coef_df['gene'].head(k).tolist()
        idx = [feat_to_idx[g] for g in top_k_genes]
        X_tr_k = X_tr[:, idx]
        X_te_k = X_te[:, idx]
        mdl = LogisticRegression(
            solver='saga', max_iter=20000,
            class_weight='balanced', random_state=BASE_SEED,
        )
        mdl.fit(X_tr_k, y_train)
        auc_k = roc_auc_score(y_test, mdl.predict_proba(X_te_k)[:, 1])
        sweep_rows.append({'k': k, 'method': 'lasso_topk', 'holdout_auc': auc_k})
        mark = "   <- TARGET_K" if k == TARGET_K else ""
        print(f"  lasso_topk   k={k:>3}   Holdout AUC = {auc_k:.4f}{mark}")

    # dodatkowy filter baseline: ANOVA F SelectKBest
    print()
    for k in K_SWEEP:
        if k > X_tr.shape[1]:
            continue
        skb = SelectKBest(f_classif, k=k).fit(X_tr, y_train)
        X_tr_f = skb.transform(X_tr)
        X_te_f = skb.transform(X_te)
        mdl = LogisticRegression(
            solver='saga', max_iter=20000,
            class_weight='balanced', random_state=BASE_SEED,
        )
        mdl.fit(X_tr_f, y_train)
        auc_f = roc_auc_score(y_test, mdl.predict_proba(X_te_f)[:, 1])
        sweep_rows.append({'k': k, 'method': 'anova_topk', 'holdout_auc': auc_f})
        mark = "   <- TARGET_K" if k == TARGET_K else ""
        print(f"  anova_topk   k={k:>3}   Holdout AUC = {auc_f:.4f}{mark}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv('lasso_baseline_k_sweep.csv', index=False)

    # wyciagnij numery dla TARGET_K do podsumowania
    tgt = sweep_df[sweep_df['k'] == TARGET_K].set_index('method')['holdout_auc'].to_dict()

    # ── 6. Podsumowanie ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PODSUMOWANIE BASELINE")
    print('='*60)
    print(f"  #genow wejsciowych po Constant drop   : {len(coefs)}")
    print(f"  #niezerowych wsp. po LASSO            : {n_nonzero}")
    print(f"  CV AUC (train, {CV_FOLDS}-fold)             : {best_cv_auc:.4f}")
    print(f"  Holdout AUC (pelne LASSO)             : {holdout_auc:.4f}")
    print(f"  Holdout AUC (plain LogReg L2 ref)     : {plain_auc:.4f}")
    print(f"\n  --- fair-compare przy k={TARGET_K} genow ---")
    if 'lasso_topk' in tgt:
        print(f"  Holdout AUC (top-{TARGET_K} LASSO + LogReg)   : {tgt['lasso_topk']:.4f}")
    if 'anova_topk' in tgt:
        print(f"  Holdout AUC (top-{TARGET_K} ANOVA + LogReg)   : {tgt['anova_topk']:.4f}")
    print("\nPorownaj z wynikami pipeline'u stability + RFE (12 genow):")
    print(f"  jesli top-{TARGET_K} LASSO/ANOVA bije stability+RFE,")
    print(f"  to caly skomplikowany pipeline nie dodaje wartosci na tym datasecie.")


if __name__ == '__main__':
    main()
