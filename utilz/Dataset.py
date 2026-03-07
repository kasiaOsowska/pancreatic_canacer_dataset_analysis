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

    @staticmethod
    def load(path):
        X = pd.read_csv(os.path.join(path, 'X.csv'), index_col=0)
        meta = pd.read_csv(os.path.join(path, 'meta.csv'), index_col=0)
        y = pd.read_csv(os.path.join(path, 'y.csv'), index_col=0).squeeze()
        return Dataset(X=X, meta=meta, y=y)

    def get_pathway_scores(self, library='Reactome_2022', cache_dir=None,
                           min_size=5, max_size=500, threads=4):
        if cache_dir and os.path.exists(os.path.join(cache_dir, 'X.csv')):
            print(f"Wczytywanie z cache: {cache_dir}")
            return Dataset.load(cache_dir)

        mg = mygene.MyGeneInfo()
        result = mg.querymany(
            self.X.columns.tolist(),
            scopes='ensembl.gene',
            fields='symbol',
            species='human',
            as_dataframe=True
        )
        ensembl_to_symbol = result['symbol'].dropna().to_dict()

        X_sym = self.X.rename(columns=ensembl_to_symbol)
        X_sym = X_sym[[c for c in X_sym.columns if not c.startswith('ENSG')]]
        X_sym.columns = X_sym.columns.str.upper()
        X_sym = X_sym.T.groupby(level=0).mean().T

        gene_sets = gp.get_library(library)

        ss = gp.ssgsea(
            data=X_sym.T,
            gene_sets=gene_sets,
            sample_norm_method='rank',
            no_plot=True,
            outdir=None,
            min_size=min_size,
            max_size=max_size,
            threads=threads
        )

        scores_df = ss.res2d.pivot_table(index='Name', columns='Term', values='NES')
        scores_df = scores_df.loc[self.X.index]

        ds_pathways = Dataset(X=scores_df, meta=self.meta, y=self.y)

        if cache_dir:
            ds_pathways.save(cache_dir)
            print(f"Zapisano cache: {cache_dir}")

        return ds_pathways

    def get_train_test_valid_split(self, X, y, test_size=0.25, valid_size=0.25, random_state=2137):
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(test_size + valid_size),
            random_state=random_state,
            stratify=y
        )

        relative_valid_size = valid_size / (test_size + valid_size)

        X_test, X_valid, y_test, y_valid = train_test_split(
            X_temp, y_temp,
            test_size=relative_valid_size,
            random_state=random_state,
            stratify=y_temp
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