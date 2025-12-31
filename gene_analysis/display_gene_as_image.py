import numpy as np
import matplotlib.pyplot as plt
from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY, CANCER


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic_filtered.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
X = ds.X.values


def display_genes_as_image(index):
    example = X[index]
    n_genes = len(example)
    print(f"Number of genes: {n_genes}")

    def find_best_shape(n):
        sqrt_n = int(np.sqrt(n))
        for i in range(sqrt_n, 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n

    rows, cols = find_best_shape(n_genes)
    print(f"Image shape: {rows} × {cols} = {rows * cols}")

    example_image = example.reshape(rows, cols)

    plt.figure(figsize=(10, 7))
    plt.imshow(example_image, cmap='hot', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Expression Level', shrink=0.8)
    plt.title(f'Gene Expression Pattern {n_genes} genes {rows}×{cols} image \n Label: {ds.y[index]}')
    plt.xlabel(f'Genes (columns: {cols})')
    plt.ylabel(f'Genes (rows: {rows})')
    plt.tight_layout()
    plt.show()


disease_index = np.where(ds.y == DISEASE)[0][0]
display_genes_as_image(disease_index)

healthy_index = np.where(ds.y == HEALTHY)[0][0]
display_genes_as_image(healthy_index)

cancer_index = np.where(ds.y == CANCER)[0][0]
display_genes_as_image(cancer_index)