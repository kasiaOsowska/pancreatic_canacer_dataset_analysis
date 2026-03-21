import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, BatchNorm


class PancreaticGAT(torch.nn.Module):
    """
    Graph Attention Network do binarnej klasyfikacji rak/zdrowy.

    Architektura:
        GATConv(1 → H, heads=4) → BN → ReLU
        GATConv(4H → H, heads=4) → BN → ReLU
        GATConv(4H → H, heads=1) → BN → ReLU
        GlobalPool (mean + max concat)
        FC: 2H → H/2 → 1  →  Sigmoid
    """
    def __init__(self, hidden: int = 64, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(1,          hidden, heads=4, dropout=dropout)
        self.bn1   = BatchNorm(hidden * 4)

        self.conv2 = GATConv(hidden * 4, hidden, heads=4, dropout=dropout)
        self.bn2   = BatchNorm(hidden * 4)

        self.conv3 = GATConv(hidden * 4, hidden, heads=1, dropout=dropout, concat=False)
        self.bn3   = BatchNorm(hidden)

        # mean + max pooling concat → 2*hidden
        self.fc1   = torch.nn.Linear(2 * hidden, hidden // 2)
        self.fc2   = torch.nn.Linear(hidden // 2, 1)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.bn3(self.conv3(x, edge_index)))

        # Readout: mean + max pool → concat
        x_mean = global_mean_pool(x, batch)   # [B, hidden]
        x_max  = global_max_pool(x, batch)    # [B, hidden]
        x      = torch.cat([x_mean, x_max], dim=1)  # [B, 2*hidden]

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.fc2(x)).squeeze(1)   # [B]

    def get_attention_weights(self, x, edge_index):
        """Do interpretowalności: zwraca wagi atencji per krawędź."""
        _, (edge_idx, alpha1) = self.conv1(x, edge_index, return_attention_weights=True)
        return edge_idx, alpha1
