import pandas as pd
import mygene
import gseapy as gp

from utilz.Dataset import load_dataset


meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
# ╔════════════════════════════════════════════╗
# ║  ZMIEŃ TĘ LINIĘ NA SWOJĄ ŚCIEŻKĘ:        ║
# ╚════════════════════════════════════════════╝
df = ds.X.copy()

# ---------- ENSG → Symbol ----------
df.columns = [c.split('.')[0] for c in df.columns]
mg = mygene.MyGeneInfo()
res = mg.querymany(list(df.columns), scopes='ensembl.gene',
                   fields='symbol', species='human', returnall=True)
ensg2sym = {h['query']: h['symbol'] for h in res['out'] if 'symbol' in h}
print(f"Zmapowano {len(ensg2sym)}/{len(df.columns)} genów")

# ---------- Pobierz Reactome ----------
reactome = gp.get_library(name='Reactome_2022', organism='Human')
print(f"Reactome: {len(reactome)} szlaków")

# ---------- Odwróć: symbol → lista szlaków ----------
sym2pathways = {}
for pathway, genes in reactome.items():
    for gene in genes:
        sym2pathways.setdefault(gene, []).append(pathway)

# ---------- Zbuduj DataFrame ----------
rows = []
for ensg, sym in ensg2sym.items():
    if sym in sym2pathways:
        for pw in sym2pathways[sym]:
            rows.append({'ensg_id': ensg, 'symbol': sym, 'pathway': pw})
    else:
        rows.append({'ensg_id': ensg, 'symbol': sym, 'pathway': 'brak w Reactome'})

result = pd.DataFrame(rows)
result.to_csv('gene_reactome.csv', index=False)

print(f"\n{result['ensg_id'].nunique()} genów, {result['pathway'].nunique()} szlaków")
print(f"\nTop 20 szlaków wg liczby genów:")
print(result[result['pathway'] != 'brak w Reactome']
      .groupby('pathway')['symbol'].nunique()
      .sort_values(ascending=False).head(20).to_string())
print(f"\nZapisano: gene_reactome.csv")