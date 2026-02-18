import numpy as np
import matplotlib.pyplot as plt
from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY, CANCER


meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
X = ds.X.values

import numpy as np
import matplotlib.pyplot as plt

X = ds.X.values

def find_best_shape(n):
    sqrt_n = int(np.sqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n

def show_group_means_one_figure(labels, names, sort=True):
    means = []
    counts = []
    for lab in labels:
        mask = (ds.y == lab)
        counts.append(int(mask.sum()))
        means.append(X[mask].mean(axis=0))

    n_genes = means[0].size
    rows, cols = find_best_shape(n_genes)

    if sort:
        global_mean = X.mean(axis=0)
        order = np.argsort(global_mean)[::-1]
        means = [m[order] for m in means]

    vmin = min(m.min() for m in means)
    vmax = max(m.max() for m in means)

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), constrained_layout=True)

    if len(labels) == 1:
        axes = [axes]

    images = []
    for ax, m, name, cnt in zip(axes, means, names, counts):
        img = m.reshape(rows, cols)
        im = ax.imshow(img, cmap="hot", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\n(n={cnt})")
        ax.set_xlabel("Genes")
        ax.set_ylabel("Genes")
        images.append(im)

    fig.colorbar(images[0], ax=axes, shrink=0.85, label="Mean Expression Level")
    plt.show()

show_group_means_one_figure(
    labels=[HEALTHY, DISEASE, CANCER],
    names=["HEALTHY", "DISEASE", "CANCER"],
    sort=True
)
