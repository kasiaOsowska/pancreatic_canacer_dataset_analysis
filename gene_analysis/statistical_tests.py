from scipy.stats import mannwhitneyu
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from utilz.Dataset import load_dataset
from utilz.constans import *
from utilz.helpers import plot_scatter_boxplot, plot_roc_curve

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

def calculate_statistical_tests(gene_name):
    X = ds.X[gene_name]
    y = ds.y
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
    plot_roc_curve(X, y_encoded, gene_name + " disease combined with healthy")
    plot_scatter_boxplot(X, y, gene_name)


"""
X_TP53 = ds.X[TP53]
calculate_statistical_tests("TP53")

X_SMAD4 = ds.X[SMAD4]
calculate_statistical_tests("SMAD4")
X_KRAS = ds.X[KRAS]
calculate_statistical_tests("KRAS")

X_ARL2 = ds.X[ARL2]
calculate_statistical_tests("ARL2")
X_BCAP31 = ds.X[BCAP31]
calculate_statistical_tests("BCAP31")
X_CFL1= ds.X[CFL1]
calculate_statistical_tests("CFL1")
X_MYL9 = ds.X[MYL9]
calculate_statistical_tests("MYL9")

"""
# worst shap
calculate_statistical_tests("ENSG00000131828")

#best shap
calculate_statistical_tests("ENSG00000109814")



