import os
import gseapy as gp
import mygene

from utilz.preprocessing_utilz import *
from utilz.helpers import *
from utilz.Dataset import load_dataset, Dataset


def load(path):
    X = pd.read_csv(os.path.join(path, 'X.csv'), index_col=0)
    meta = pd.read_csv(os.path.join(path, 'meta.csv'), index_col=0)
    y = pd.read_csv(os.path.join(path, 'y.csv'), index_col=0).squeeze()
    return Dataset(X=X, meta=meta, y=y)

def get_pathway_scores(ds, library='Reactome_2022', cache_dir=None,
                       min_size=5, max_size=500, threads=4):
    if cache_dir and os.path.exists(os.path.join(cache_dir, 'X.csv')):
        print(f"Wczytywanie z cache: {cache_dir}")
        return load(cache_dir)

    mg = mygene.MyGeneInfo()
    result = mg.querymany(
        ds.X.columns.tolist(),
        scopes='ensembl.gene',
        fields='symbol',
        species='human',
        as_dataframe=True
    )
    ensembl_to_symbol = result['symbol'].dropna().to_dict()

    X_sym = ds.X.rename(columns=ensembl_to_symbol)
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
    scores_df = scores_df.loc[ds.X.index]

    ds_pathways = Dataset(X=scores_df, meta=ds.meta, y=ds.y)

    if cache_dir:
        ds_pathways.save(cache_dir)
        print(f"Zapisano cache: {cache_dir}")

    return ds_pathways