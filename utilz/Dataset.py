# structure:
# dataset.X : DataFrame (rows: samples x columns: features(genes))
# dataset.meta : DataFrame (samples x metadata)
# dataset.y : Series (labels)
import pandas as pd
from sklearn.model_selection import train_test_split
from utilz.constans import DISEASE, HEALTHY, CANCER

class Dataset:
    def __init__(self, X, meta, y=None, ):
        self.X = X
        self.meta = meta
        self.y = y
        self.age = self.meta['Age']
        self.sex =  self.meta['Sex']

    def get_train_test_valid_split(self, X, y, test_size=0.25, valid_size=0.25, random_state=2137):
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(test_size + valid_size), 
            random_state=random_state, 
            stratify=y
        )
        
        # Calculate relative size for valid_split from the remaining part
        relative_valid_size = valid_size / (test_size + valid_size)
        
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_temp, y_temp, 
            test_size=relative_valid_size, 
            random_state=random_state, 
            stratify=y_temp
        )
        
        return X_train, X_test, X_valid, y_train, y_test, y_valid

def load_dataset(path_csv, path_xlsx, label_col=None):
    df_features =  pd.read_csv(path_csv, sep=";", decimal=",", index_col=0)
    df_features = df_features.T
    df_metadata = pd.read_excel(path_xlsx, index_col=0)
    intersect = df_features.index.intersection(df_metadata.index)

    if len(df_features.index) != len(intersect):
        print(f"[INFO] skipped {len(df_features.index) - len(intersect)} probs due to missing metadata")

    df_features = df_features.loc[intersect].sort_index()
    meta = df_metadata.loc[intersect].sort_index()

    y = meta[label_col] if label_col is not None else meta["Group"]

    return Dataset(X=df_features, meta=meta, y=y)