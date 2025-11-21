# structure:
# dataset.X : DataFrame (rows: samples x columns: features)
# dataset.meta : DataFrame (samples x metadata)
# dataset.y : Series (labels, optional)
import pandas as pd
from utilz import *

class Dataset:
    def __init__(self, X, meta, y=None, ):
        self.X = X
        self.meta = meta
        self.y = y

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

    def training_split(self, test_size=0.2, random_state=42):
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
        for label in [HEALTHY, DISEASE, CANCER]:
            X_class = self.X[self.y == label]
            y_class = self.y[self.y == label]
            X_train = X_class.sample(frac=1 - test_size, random_state=random_state)
            X_test = X_class.drop(X_train.index)
            y_train = y_class.loc[X_train.index]
            y_test = y_class.drop(X_train.index)

            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)

        X_train = pd.concat(X_train_list).sort_index()
        X_test = pd.concat(X_test_list).sort_index()
        y_train = pd.concat(y_train_list).sort_index()
        y_test = pd.concat(y_test_list).sort_index()
        return X_train, y_train, X_test, y_test

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