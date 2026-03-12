import os
import pandas as pd
import mygene
import gseapy as gp
from sklearn.model_selection import train_test_split
from utilz.constans import DISEASE, HEALTHY, CANCER


class Dataset:
    def __init__(self, X, meta, y=None):
        self.X = X
        self.meta = meta
        self.y = y
        self.age = self.meta['Age']
        self.sex = self.meta['Sex']

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.X.to_csv(os.path.join(path, 'X.csv'))
        self.meta.to_csv(os.path.join(path, 'meta.csv'))
        self.y.to_csv(os.path.join(path, 'y.csv'))

    def _get_strata(self, X, y):
        meta = self.meta

        df = pd.DataFrame(index=X.index)
        df["y"] = y
        df["sex"] = meta["Sex"]
        df["age_group"] = pd.qcut(meta["Age"], q=3, labels=["young", "mid", "old"])
        df = df.dropna(subset=["y", "sex", "age_group"])

        X = X.loc[df.index]
        y = y.loc[df.index]

        strata = df[["y", "sex", "age_group"]].astype(str).agg("_".join, axis=1)

        counts = strata.value_counts()
        valid = counts[counts >= 2].index
        mask = strata.isin(valid)

        X = X[mask]
        y = y[mask]
        strata = strata[mask]
        return X, y, strata

    def get_train_test_valid_split(self, X, y, test_size=0.25, valid_size=0.25, random_state=2137):

        X, y, strata = self._get_strata(X, y)

        X_train, X_temp, y_train, y_temp, strata_train, strata_temp = train_test_split(
            X, y, strata,
            test_size=(test_size + valid_size),
            random_state=random_state,
            stratify=strata
        )

        relative_valid_size = valid_size / (test_size + valid_size)

        X_temp, y_temp, strata = self._get_strata(X_temp, y_temp)

        X_test, X_valid, y_test, y_valid = train_test_split(
            X_temp, y_temp,
            test_size=relative_valid_size,
            random_state=random_state,
            stratify=strata
        )

        return X_train, X_test, X_valid, y_train, y_test, y_valid


def load_dataset(path_csv, path_xlsx, label_col=None):
    df_features = pd.read_csv(path_csv, sep=";", decimal=",", index_col=0)
    df_features = df_features.T
    df_metadata = pd.read_excel(path_xlsx, index_col=0)
    intersect = df_features.index.intersection(df_metadata.index)

    if len(df_features.index) != len(intersect):
        print(f"[INFO] skipped {len(df_features.index) - len(intersect)} probs due to missing metadata")

    df_features = df_features.loc[intersect].sort_index()
    meta = df_metadata.loc[intersect].sort_index()

    y = meta[label_col] if label_col is not None else meta["Group"]

    return Dataset(X=df_features, meta=meta, y=y)