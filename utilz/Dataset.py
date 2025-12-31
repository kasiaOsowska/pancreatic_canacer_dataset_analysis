# structure:
# dataset.X : DataFrame (rows: samples x columns: features(genes))
# dataset.meta : DataFrame (samples x metadata)
# dataset.y : Series (labels)
import pandas as pd
from utilz.constans import DISEASE, HEALTHY, CANCER

class Dataset:
    def __init__(self, X, meta, y=None, ):
        self.X = X
        self.meta = meta
        self.y = y
        self.age = self.meta['Age']

        sex = self.meta['Sex']
        # some of the samples have sex set to n.a.
        mask_f = sex == 'F'
        mask_m = sex == 'M'
        X_female = self.X.loc[mask_f]
        y_female = self.y.loc[mask_f]
        X_male = self.X.loc[mask_m]
        y_male = self.y.loc[mask_m]

        self.X_female = X_female
        self.y_female = y_female
        self.X_male = X_male
        self.y_male = y_male




def load_dataset(path_csv, path_xlsx, label_col=None):
    df_features =  pd.read_csv(path_csv, sep=";", decimal=",", index_col=0)
    df_features = df_features.T
    df_metadata = pd.read_excel(path_xlsx, index_col=0)
    intersect = df_features.index.intersection(df_metadata.index)

    if len(df_features.index) != len(intersect):
        print(f"[INFO] skipped {len(df_features.index) - len(intersect)} probs due to missing metadata")

    df_features = df_features.loc[intersect].sort_index()
    meta = df_metadata.loc[intersect].sort_index()

    y = meta[label_col] if label_col is not None else "Group"

    return Dataset(X=df_features, meta=meta, y=y)