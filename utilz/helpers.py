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

            for key in ["TP", "FP", "FN", "TN"]:
                print(f"\n--- {key} samples metadata ---")
                for idx in results[cls][key]:
                    sample_meta = dataset.meta.loc[idx]
                    print(f"{key} - Sample ID: {idx}, Metadata:")
                    print(sample_meta["Group"], sample_meta["Sex"], sample_meta["Age"], sample_meta["Stage"])
                    print("---")

    return


def plot_pca(X, y_encoded, n_compontns, le):
    viz_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=n_compontns, svd_solver="full", random_state=42))
    ])

    X_train_pca = viz_pipe.fit_transform(X)
    evr = viz_pipe.named_steps["pca"].explained_variance_ratio_
    classes = le.classes_
    y_test_int = y_encoded.values

    plt.figure(figsize=(7, 6))
    for i in range(n_compontns):
        for j in range(i + 1, n_compontns):
            for k, cls in enumerate(classes):
                mask = (y_test_int == k)
                plt.scatter(
                    X_train_pca[mask, i], X_train_pca[mask, j],
                    label=str(cls), alpha=0.75, s=45
                )

            plt.xlabel(f"PC{i} ({evr[i] * 100:.1f}% wariancji)")
            plt.ylabel(f"PC{j} ({evr[j] * 100:.1f}% wariancji)")
            plt.title("PCA")
            plt.legend(title="Klasa")
            plt.grid(True, linestyle="--", alpha=0.3)
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


def plot_roc_curve(X, y, gene_name):
    le = LabelEncoder()
    y_combined = y.replace({DISEASE: HEALTHY})
    y_encoded = pd.Series(le.fit_transform(y_combined), index=y_combined)

    fpr, tpr, thresholds = roc_curve(y_encoded, X)

    auc = roc_auc_score(y_encoded, X)
    print(f"ROC AUC = {auc:.3f}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate (1 - specificity)")
    plt.ylabel("True Positive Rate (sensitivity)")
    plt.title("ROC curve for " + gene_name + "DISEASE merged with HEALTHY")
    plt.legend()
    plt.grid(alpha=0.1)
    plt.savefig(f"graphics/{gene_name}_fpr_tpr.png")
    plt.show()

