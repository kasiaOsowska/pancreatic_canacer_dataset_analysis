import os
import random

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY
from utilz.helpers import plot_roc_curve
from utilz.preprocessing_utilz import *

# --- reproducibility ---
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- data loading ---
meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)
sex_numeric = ds.sex.map({"F": 0, "M": 1})

X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25))

pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaReductor', AnovaReductor(percentile=80)),
    ('MeanExpressionReductor', MeanExpressionReductor(percentile=25)),
    ('AgeBiasReductor',  CovariatesBiasReductor(covariate=ds.age)),
    ('SexBiasReductor',  CovariatesBiasReductor(covariate=sex_numeric)),
    ('scaler', StandardScaler())
])

X_train = pipeline.fit_transform(X_train, y_train)
X_valid = pipeline.transform(X_valid)
X_test = pipeline.transform(X_test)


n_features = X_train.shape[1]
print(f"Features: {n_features}, Train: {len(X_train)}, Test: {len(X_test)}, Valid: {len(X_valid)}")


# --- datasets & loaders ---
class GeneDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)  # (N, 1, genes)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


batch_size = 32
train_loader = torch.utils.data.DataLoader(GeneDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(GeneDataset(X_valid, y_valid), batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(GeneDataset(X_test, y_test), batch_size=batch_size)


# --- model ---
class CancerCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(4),

            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


model = CancerCNN(n_features).to(device)
print(model)

# --- class weights ---
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
print(f"Class ratio neg/pos: {n_neg}/{n_pos}, pos_weight: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)

# --- training ---
epochs = 100
patience = 10
best_val_loss = float("inf")
patience_counter = 0
best_state = None
train_losses, val_losses = [], []

for epoch in range(epochs):
    # train
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(y_batch)
    train_loss = running_loss / len(train_loader.dataset)

    # validate
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            val_running += criterion(logits, y_batch).item() * len(y_batch)
    val_loss = val_running / len(valid_loader.dataset)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr:.1e}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)

# --- loss curves ---
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="valid")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- evaluation ---
model.eval()
all_logits, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        all_logits.append(logits.cpu())
        all_labels.append(y_batch)

all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels).numpy()
y_probs = torch.sigmoid(all_logits).numpy()
y_pred = (y_probs > 0.5).astype(int)

print("\n" + classification_report(all_labels, y_pred, target_names=le.classes_))

cm = confusion_matrix(all_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

plot_roc_curve(y_probs, all_labels, "CNN PyTorch")
