import pandas as pd
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

HEALTHY = "Asymptomatic controls"
DISEASE = "Pancreatic diseases"
CANCER = "Pancreatic cancer"

# GENES KNOWN TO CORRELATE WITH PANCREATIC CANCER FROM ARTICLES
KRAS = "ENSG00000133703"
TP53 = "ENSG00000141510"
SMAD4 = "ENSG00000141646"

# MINIMAL P-VALUE GENES FOUND
BCAP31 = "ENSG00000185825"
ARL2 = "ENSG00000213465"
CFL1 = "ENSG00000172757"
MYL9 = "ENSG00000101335"


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
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(7, 6))
    for i in range(n_compontns):
        for j in range(i + 1, n_compontns):
            for k, cls in enumerate(classes):
                mask = (y_test_int == k)
                plt.scatter(
                    X_train_pca[mask, i], X_train_pca[mask, j],
                    label=str(cls), alpha=0.75, s=45
                )

            plt.xlabel(f"PC{i} ({evr[0] * 100:.1f}% wariancji)")
            plt.ylabel(f"PC{j} ({evr[1] * 100:.1f}% wariancji)")
            plt.title("PCA")
            plt.legend(title="Klasa")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.show()
