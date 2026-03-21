from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import pandas as pd

from utilz.constans import HEALTHY, DISEASE, CANCER

def show_report(y_pred, y_test_encoded, dataset, le):
    y_true = y_test_encoded
    y_pred_s = pd.Series(y_pred, index=y_true.index, name="y_pred")

    eval_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_s})
    class_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Mapowanie klas:", class_map)

    results = {}

    for cls, cls_code in class_map.items():
        if len(class_map) == 2 and cls_code != 0:
            TP_idx = eval_df.index[(eval_df.y_true == cls_code) & (eval_df.y_pred == cls_code)]
            FP_idx = eval_df.index[(eval_df.y_true != cls_code) & (eval_df.y_pred == cls_code)]
            FN_idx = eval_df.index[(eval_df.y_true == cls_code) & (eval_df.y_pred != cls_code)]
            TN_idx = eval_df.index[(eval_df.y_true != cls_code) & (eval_df.y_pred != cls_code)]

            results[cls] = {
                "TP": TP_idx,
                "FP": FP_idx,
                "FN": FN_idx,
                "TN": TN_idx,
            }

            for key in ["FN", "FP"]: #["TP", "FP", "FN", "TN"]:
                print(f"\n--- {key} samples metadata ---")
                for idx in results[cls][key]:
                    sample_meta = dataset.meta.loc[idx]
                    print(f"{key} - Sample ID: {idx}, Metadata:")
                    print(sample_meta["Group"], sample_meta["Sex"], sample_meta["Age"], sample_meta["Stage"])
                    print("---")

    return


from itertools import combinations
import math

def plot_pca(X, y_encoded, n_components, le):
    viz_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=n_components, svd_solver="full", random_state=42))
    ])

    X_pca = viz_pipe.fit_transform(X)
    evr = viz_pipe.named_steps["pca"].explained_variance_ratio_
    classes = le.classes_
    y_int = y_encoded.values

    pairs = list(combinations(range(n_components), 2))
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = math.ceil(n_pairs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // n_cols][idx % n_cols]
        for k, cls in enumerate(classes):
            mask = (y_int == k)
            ax.scatter(
                X_pca[mask, i], X_pca[mask, j],
                label=str(cls), alpha=0.5, s=45
            )
        ax.set_xlabel(f"PC{i+1} ({evr[i]*100:.1f}% wariancji)")
        ax.set_ylabel(f"PC{j+1} ({evr[j]*100:.1f}% wariancji)")
        ax.set_title(f"PC{i+1} vs PC{j+1}")
        ax.legend(title="Klasa", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
    for idx in range(n_pairs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("PCA – wszystkie pary składowych", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

def plot_scatter_boxplot(X, y, gene_name):
    unique_groups = np.unique(y)
    has_disease = DISEASE in unique_groups
    has_cancer = CANCER in unique_groups

    if has_disease and has_cancer:
        disease_mask = y == DISEASE
        cancer_mask = y == CANCER
        combined_mask = disease_mask | cancer_mask
        X_combined = X[combined_mask]
        y_combined = np.full(len(X_combined), 'Disease and Cancer')
        X_plot = np.concatenate([X, X_combined])
        y_plot = np.concatenate([y, y_combined])
    else:
        X_plot = X
        y_plot = y
    plt.figure(figsize=(10, 10))

    ax = sns.boxplot(y=X_plot, x=y_plot,
                     width=0.5,
                     linewidth=2)

    sns.stripplot(y=X_plot, x=y_plot,
                  color='black',
                  alpha=0.5,
                  size=5,
                  jitter=0.2)

    for i, group in enumerate(np.unique(y_plot)):
        mask = y_plot == group
        mean_val = np.mean(X_plot[mask])
        ax.hlines(mean_val, i - 0.25, i + 0.25,
                  colors='black', linewidth=2, linestyle = '--',
                  label='Średnia' if i == 0 else '')

    plt.title(f'Analiza ekspresji genu: {gene_name}',
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Poziom ekspresji', fontsize=12)
    plt.xlabel('Etykieta', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(X, y, title):
    fpr, tpr, thresholds = roc_curve(y, X)
    auc = roc_auc_score(y, X)
    print(f"ROC AUC = {auc:.3f}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate (1 - specificity)")
    plt.ylabel("True Positive Rate (sensitivity)")
    plt.title("ROC curve for " + title)
    plt.legend()
    plt.grid(alpha=0.1)
    plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(y_proba, y_true, title):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()   # random classifier = % klasy pozytywnej
    print(f"Average Precision = {ap:.3f}")

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
    plt.axhline(baseline, linestyle="--", color="gray",
                label=f"Random (baseline = {baseline:.3f})")
    plt.xlabel("Recall (sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title("Precision-Recall curve for " + title)
    plt.legend()
    plt.grid(alpha=0.1)
    plt.show()