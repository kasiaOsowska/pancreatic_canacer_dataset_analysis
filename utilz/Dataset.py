import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utilz.constans import CANCER


class Dataset:
    def __init__(self, X, meta, y=None):
        self.X = X
        self.meta = meta
        self.y = y
        self.age = self.meta['Age']
        self.sex = self.meta['Sex']

    def _get_strata(self, X, y):
        meta = self.meta.loc[X.index]

        df = pd.DataFrame(index=X.index)
        df["y"] = y
        df["sex"] = meta["Sex"]
        df["age_group"] = (
            pd.qcut(meta["Age"], q=3, labels=["young", "mid", "old"], duplicates="drop")
            .astype(object)
            .fillna("unknown")
        )
        df["stage"] = meta["Stage"].replace("NA", "none")
        stage = meta["Stage"].replace("NA", "none").fillna("none")
        stage = stage.replace({"I": "I_II", "II": "I_II"})
        df["stage"] = stage
        print(df["stage"].value_counts())

        X = X.loc[df.index]
        y = y.loc[df.index]

        strata = df[["y", "sex", "age_group", "stage"]].astype(str).agg("_".join, axis=1)

        counts = strata.value_counts()
        valid = counts[counts >= 2].index
        mask = strata.isin(valid)

        X = X[mask]
        y = y[mask]
        strata = strata[mask]
        return X, y, strata

    def get_train_test_valid_split(self, X, y, test_size=0.25, valid_size=0.25, random_state=2137, return_valid=True):
        X, y, strata = self._get_strata(X, y)

        X_train, X_temp, y_train, y_temp, strata_train, strata_temp = train_test_split(
            X, y, strata,
            test_size=(test_size + valid_size),
            random_state=random_state,
            stratify=strata
        )

        if not return_valid:
            return X_train, X_temp, y_train, y_temp

        relative_valid_size = valid_size / (test_size + valid_size)

        X_temp, y_temp, strata = self._get_strata(X_temp, y_temp)

        X_test, X_valid, y_test, y_valid = train_test_split(
            X_temp, y_temp,
            test_size=relative_valid_size,
            random_state=random_state,
            stratify=strata
        )
        _assert_no_leakage(
            X_train.index, X_test.index, X_valid.index,
            names=["Train", "Test", "Valid"]
        )

        return X_train, X_test, X_valid, y_train, y_test, y_valid


def load_dataset(path_csv, path_xlsx, label_col=None, separate_stage_iv = False):
    df_features = pd.read_csv(path_csv, sep=";", decimal=",", index_col=0)
    df_features = df_features.T
    df_metadata = pd.read_excel(path_xlsx, index_col=0)
    intersect = df_features.index.intersection(df_metadata.index)

    if len(df_features.index) != len(intersect):
        print(f"[INFO] skipped {len(df_features.index) - len(intersect)} probs due to missing metadata")

    df_features = df_features.loc[intersect].sort_index()
    meta = df_metadata.loc[intersect].sort_index()

    y = meta[label_col] if label_col is not None else meta["Group"]

    if separate_stage_iv:
        y = y.mask((y == CANCER) & (meta["Stage"] == "IV"), "cancer_IV")

    return Dataset(X=df_features, meta=meta, y=y)

def _assert_no_leakage(*splits: pd.Index, names: list[str] = None) -> None:
    if names is None:
        names = [f"Split_{i}" for i in range(len(splits))]

    ok = True
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = splits[i].intersection(splits[j])
            if len(overlap) > 0:
                print(f"LEAKAGE: {names[i]} ∩ {names[j]} = {len(overlap)}")
                print(f"   Indexes: {overlap.tolist()}")
                ok = False
            else:
                print(f"{names[i]} ∩ {names[j]} = 0")

    if ok:
        print("\n[ASSERTION PASSED] No leakage detected between splits.")
    else:
        raise AssertionError("LEAKAGE")