import numpy as np
import pandas as pd
import torch
import mygene
import gseapy as gp
from itertools import combinations
from pathlib import Path
from torch_geometric.data import Data

from .constans import CANCER


def _build_gene_pathway_mapping(
    genes: list[str],
    cache_path: Path,
    library: str = "Reactome_2022",
) -> pd.DataFrame:
    """
    Buduje lub wczytuje mapowanie gen→pathway.
    Przy pierwszym uruchomieniu pobiera dane z mygene + gseapy i zapisuje CSV.
    Przy kolejnych wczytuje z cache.
    """
    csv_path = cache_path / "gene_reactome.csv"

    if csv_path.exists():
        print(f"[Reactome] Wczytywanie mapowania z cache: {csv_path}")
        return pd.read_csv(csv_path)

    print("[Reactome] Budowanie mapowania gen→pathway (pierwsze uruchomienie)...")

    # ENSG → symbol
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        genes, scopes='ensembl.gene',
        fields='symbol', species='human', returnall=True
    )
    ensg2sym = {h['query']: h['symbol'] for h in res['out'] if 'symbol' in h}
    print(f"[Reactome] Zmapowano {len(ensg2sym)}/{len(genes)} genów na symbole")

    # Reactome pathways via gseapy
    reactome = gp.get_library(name=library, organism='Human')
    print(f"[Reactome] {len(reactome)} szlaków w {library}")

    # symbol → lista pathway
    sym2pathways: dict[str, list[str]] = {}
    for pathway, pathway_genes in reactome.items():
        for gene in pathway_genes:
            sym2pathways.setdefault(gene, []).append(pathway)

    # Buduj DataFrame
    rows = []
    for ensg, sym in ensg2sym.items():
        if sym in sym2pathways:
            for pw in sym2pathways[sym]:
                rows.append({'ensg_id': ensg, 'symbol': sym, 'pathway': pw})

    df = pd.DataFrame(rows)

    cache_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[Reactome] Zapisano mapowanie: {csv_path}")
    print(f"[Reactome] {df['ensg_id'].nunique()} genów, {df['pathway'].nunique()} szlaków")

    return df


def load_reactome_edges(
    genes: list[str],
    cache_path: str | None = None,
    min_pathway_size: int = 5,
    max_pathway_size: int = 300,
) -> tuple[torch.Tensor, list[str]]:
    """
    Zwraca edge_index i posortowaną listę genów (węzłów).
    Krawędź istnieje między dwoma genami jeśli należą do >=1 wspólnego pathway Reactome.

    Przy pierwszym uruchomieniu pobiera mapowanie gen→pathway i buduje graf.
    Przy kolejnych wczytuje z cache.

    Returns:
        edge_index : LongTensor [2, E]  (nieskierowany)
        gene_list  : list[str]          (mapowanie idx -> ensembl_id)
    """
    cache = Path(cache_path) if cache_path else Path("cache/reactome_graph")

    # Sprawdź cache grafu (walidacja: czy lista genów się zgadza)
    if (cache / "edge_index.pt").exists() and (cache / "gene_list.csv").exists():
        cached_genes = set(pd.read_csv(cache / "gene_list.csv", index_col=0)["gene"].tolist())
        input_genes = set(genes)
        if cached_genes.issubset(input_genes):
            print(f"[Reactome] Wczytywanie grafu z cache ({len(cached_genes)} węzłów)...")
            edge_index = torch.load(cache / "edge_index.pt", weights_only=True)
            gene_list = sorted(cached_genes)
            return edge_index, gene_list
        else:
            print("[Reactome] Cache nieaktualny (zmieniona lista genów), przebudowuję...")

    # Buduj/wczytaj mapowanie gen→pathway (pełne, dla wszystkich genów)
    mapping = _build_gene_pathway_mapping(genes, cache)

    # Filtruj do aktualnie przekazanych genów
    mapping = mapping[mapping["ensg_id"].isin(genes)]

    # Filtruj pathway po rozmiarze (po zawężeniu do aktualnych genów)
    pathway_sizes = mapping.groupby("pathway")["ensg_id"].nunique()
    valid_pathways = pathway_sizes[
        (pathway_sizes >= min_pathway_size) & (pathway_sizes <= max_pathway_size)
    ].index
    mapping = mapping[mapping["pathway"].isin(valid_pathways)]

    gene_list = sorted(mapping["ensg_id"].unique().tolist())
    gene_idx = {g: i for i, g in enumerate(gene_list)}

    print(f"[Reactome] {len(gene_list)} genów | {mapping['pathway'].nunique()} szlaków")

    # Budowanie krawędzi przez co-occurrence w pathway
    edges_set: set[tuple[int, int]] = set()
    for _, grp in mapping.groupby("pathway"):
        gs = [gene_idx[g] for g in grp["ensg_id"].unique()]
        for a, b in combinations(gs, 2):
            edges_set.add((min(a, b), max(a, b)))

    if not edges_set:
        raise ValueError("Brak krawędzi! Sprawdź ID genów i mapowanie pathway.")

    src, dst = zip(*edges_set)
    edge_index = torch.tensor(
        [list(src) + list(dst), list(dst) + list(src)], dtype=torch.long
    )

    # Zapisz cache grafu
    cache.mkdir(parents=True, exist_ok=True)
    torch.save(edge_index, cache / "edge_index.pt")
    pd.DataFrame({"gene": gene_list}).to_csv(cache / "gene_list.csv")
    print(f"[Reactome] Zapisano cache grafu -> {cache}")
    print(f"[Reactome] Graf: {len(gene_list)} węzłów | {edge_index.shape[1]} krawędzi")

    return edge_index, gene_list


def build_graph_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    cache_dir: str | None = "cache/reactome_graph",
    log_transform: bool = True,
    max_pathway_size: int = 100,
) -> tuple[list[Data], list[str]]:
    """
    Zamienia macierz pacjenci×geny na listę grafów PyG.
    Jeden graf = jeden pacjent. Węzły = geny, krawędzie = Reactome co-pathway.

    Args:
        X: DataFrame [samples x genes] z ENSG ID jako kolumny
        y: Series z etykietami (np. "Pancreatic cancer" / "Asymptomatic controls")
        cache_dir: ścieżka do cache grafu
        log_transform: czy stosować log1p
        max_pathway_size: maks. rozmiar pathway do budowy krawędzi

    Returns:
        graphs: lista obiektów Data (PyG)
        gene_list: lista genów (węzłów) w grafie
    """
    # Wyczyść wersje z ENSG ID (np. ENSG00000123456.7 -> ENSG00000123456)
    X_clean = X.copy()
    X_clean.columns = [c.split('.')[0] for c in X_clean.columns]
    X_clean = X_clean.T.groupby(level=0).mean().T

    edge_index, gene_list = load_reactome_edges(
        genes=X_clean.columns.tolist(),
        cache_path=cache_dir,
        max_pathway_size=max_pathway_size,
    )

    # Wyrównaj kolumny do gene_list (tylko geny w Reactome)
    X_aligned = X_clean.reindex(columns=gene_list).fillna(0.0)

    if log_transform:
        X_aligned = np.log1p(X_aligned)

    # Normalizacja per-sample (z-score per pacjent)
    mu = X_aligned.mean(axis=1)
    std = X_aligned.std(axis=1).replace(0, 1)
    X_aligned = X_aligned.sub(mu, axis=0).div(std, axis=0)

    graphs = []
    for sample_id in X_aligned.index:
        expr = torch.tensor(X_aligned.loc[sample_id].values, dtype=torch.float)
        x = expr.unsqueeze(1)  # [N_genes, 1]
        y_val = 1.0 if y.loc[sample_id] == CANCER else 0.0
        label = torch.tensor([y_val], dtype=torch.float)

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            y=label,
            sample_id=sample_id,
        ))

    print(f"[GraphDataset] {len(graphs)} grafów | {len(gene_list)} węzłów | "
          f"{edge_index.shape[1]} krawędzi")
    return graphs, gene_list
