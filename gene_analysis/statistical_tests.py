from Dataset import load_dataset
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from utilz import *
from matplotlib import pyplot as plt


def calculate_statistical_tests(X, y, gene_name):
    print(f"Statistical tests for gene: {gene_name}")
    X_healthy = X[y == HEALTHY]
    X_disease = X[y == DISEASE]

    u_stat, p_val = mannwhitneyu(X_healthy, X_disease, alternative="two-sided")
    print("healthy vs disease")
    print(f"Mann-Whitney U-test: U = {u_stat:.3f}, p = {p_val:.3e}")

    print("healthy vs cancer")
    X_healthy = X[y == HEALTHY]
    X_cancer = X[y == CANCER]
    u_stat, p_val = mannwhitneyu(X_healthy, X_cancer, alternative="two-sided")
    print(f"Mann-Whitney U-test: U = {u_stat:.3f}, p = {p_val:.3e}")

    print("healthy and disease vs cancer")
    X_healthy = X[y.isin([HEALTHY, DISEASE])]
    X_cancer = X[y == CANCER]
    t_stat, p_val = mannwhitneyu(X_healthy, X_cancer, alternative="two-sided")
    print(f"Mann-Whitney U-test: U = {u_stat:.3f}, p = {p_val:.3e}")

    le = LabelEncoder()
    y_combined = y.replace({DISEASE: HEALTHY})
    y_encoded = pd.Series(le.fit_transform(y_combined), index=y_combined)

    fpr, tpr, thresholds = roc_curve(y_encoded, X)
    auc = roc_auc_score(y_encoded, X)
    print(f"ROC AUC = {auc:.3f}")

    class_map = {HEALTHY: "Healthy", DISEASE: "Disease", CANCER: "Cancer"}
    y_labels = y.map(class_map)

    plt.figure()
    plt.scatter(X, y_labels, label="Train points", alpha=0.7)
    means = X.groupby(y).mean()
    for cls, mean_val in means.items():
        class_str = class_map[cls]
        plt.plot([mean_val, mean_val], [class_str, class_str],
                 marker="|", markersize=25, color="black", linewidth=2,
                 label=f"{class_str} mean = {mean_val:.2f}")
    plt.xlabel(gene_name + " expression")
    plt.savefig(f"graphics/{gene_name}_expression_scatter.png")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate (1 - specificity)")
    plt.ylabel("True Positive Rate (sensitivity)")
    plt.title("ROC curve for " + gene_name)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"graphics/{gene_name}_fpr_tpr.png")
    plt.show()



meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

X_TP53 = ds.X[TP53]
calculate_statistical_tests(X_TP53, ds.y, "TP53")
X_SMAD4 = ds.X[SMAD4]
calculate_statistical_tests(X_SMAD4, ds.y, "SMAD4")
X_KRAS = ds.X[KRAS]
calculate_statistical_tests(X_KRAS, ds.y, "KRAS")


X_ARL2 = ds.X[ARL2]
calculate_statistical_tests(X_ARL2, ds.y, "ARL2")
X_BCAP31 = ds.X[BCAP31]
calculate_statistical_tests(X_BCAP31, ds.y, "BCAP31")
X_CFL1= ds.X[CFL1]
calculate_statistical_tests(X_CFL1, ds.y, "MAGOHB")
X_MYL9 = ds.X[MYL9]
calculate_statistical_tests(X_MYL9, ds.y, "PLD4")