import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from utilz.Dataset import load_dataset
from utilz.constans import CANCER
from utilz.preprocessing_utilz import (
    ConstantExpressionReductor,
    HighVarianceReductor,
    MeanExpressionReductor,
    CovariatesBiasReductor, MRMRReductor,
)
from gnn_utils.reactome_graph import build_graph_dataset
from models.gnn import PancreaticGAT


# ── Konfiguracja ───────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS     = 1000
LR         = 3e-4
HIDDEN     = 64
DROPOUT    = 0.4
SEED       = 2137

torch.manual_seed(SEED)
np.random.seed(SEED)

meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"
ds = load_dataset(data_path, meta_path, label_col="Group")
X_train, X_test, X_valid, y_train, y_test, y_valid = \
    ds.get_train_test_valid_split(ds.X, ds.y)
print(f"Split: Train={len(X_train)} | Val={len(X_valid)} | Test={len(X_test)}")
y_train_binary = (y_train == CANCER).astype(int)

reduction_pipeline = Pipeline([
    ('constant_expr', ConstantExpressionReductor()),
    ('high_var',      HighVarianceReductor(percentile=90)),
    ('mean_expr',     MeanExpressionReductor(percentile=25)),
    ('age_bias',      CovariatesBiasReductor(covariate=ds.age)),
   # ('MRMRReductor',      MRMRReductor(500)),
])

X_train_r = reduction_pipeline.fit_transform(X_train, y_train_binary)
X_test_r  = reduction_pipeline.transform(X_test)
X_valid_r = reduction_pipeline.transform(X_valid)
print(f"[Redukcja] {ds.X.shape[1]} -> {X_train_r.shape[1]} genów")

# ── Budowa grafów (ta sama struktura, różne cechy węzłów) ──────
cache = "cache/reactome_graph_reduced"
train_graphs, gene_list = build_graph_dataset(X_train_r, y_train, cache_dir=cache)
val_graphs,   _         = build_graph_dataset(X_valid_r, y_valid, cache_dir=cache)
test_graphs,  _         = build_graph_dataset(X_test_r,  y_test,  cache_dir=cache)

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE)

labels = [int(g.y.item()) for g in train_graphs]
print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

model     = PancreaticGAT(hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

pos_weight = torch.tensor(
    [sum(1-l for l in labels) / max(sum(labels), 1)], dtype=torch.float
).to(DEVICE)
criterion = torch.nn.BCELoss()


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, preds, targets = 0., [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(DEVICE)
            out   = model(batch.x, batch.edge_index, batch.batch)
            loss  = criterion(out, batch.y.float())

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.cpu().detach().numpy())
            targets.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(targets, preds) if len(set(targets)) > 1 else 0.5
    return avg_loss, auc


best_val_auc, best_state = 0., None

for epoch in range(1, EPOCHS + 1):
    train_loss, train_auc = run_epoch(train_loader, train=True)
    val_loss,   val_auc   = run_epoch(val_loader,   train=False)
    scheduler.step(val_loss)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | "
              f"Train loss: {train_loss:.4f}  AUC: {train_auc:.3f} | "
              f"Val   loss: {val_loss:.4f}  AUC: {val_auc:.3f}")

# ── Ewaluacja testowa ───────────────────────────────────────────
model.load_state_dict(best_state)
test_loss, test_auc = run_epoch(test_loader, train=False)
print(f"\nTest AUC: {test_auc:.4f}  |  Loss: {test_loss:.4f}")

# Raport z progiem 0.5
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        out   = model(batch.x, batch.edge_index, batch.batch)
        all_preds.extend((out > 0.5).int().cpu().numpy())
        all_targets.extend(batch.y.int().cpu().numpy())

print(classification_report(all_targets, all_preds,
                             target_names=["Healthy", "Pancreatic Cancer"]))

torch.save(best_state, "models/best_gnn.pt")
