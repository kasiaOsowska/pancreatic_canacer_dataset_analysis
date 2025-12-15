from Dataset import load_dataset
from utilz import *


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

def display_gene_as_image(gene_name):
    if gene_name not in ds.X.columns:
        print(f"Gene {gene_name} not found in dataset.")
        return

    gene_expression = ds.X[gene_name]
    plt.figure(figsize=(10, 6))
    plt.hist(gene_expression, bins=30, color='blue', alpha=0.7)
    plt.title(f'Expression Distribution of Gene: {gene_name}')
    plt.xlabel('Expression Level')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

#display_gene_as_image("ENSG00000182578")

print(ds.X[0])