"""
ape-MPNN: Attention Pooling Embedded MPNN
Champion solution of Tencent Alchemy Contest 2019 (NJU_Chem team)

Reference:
Liu, Ziteng, et al. "Transferable multi-level attention neural network for 
accurate prediction of quantum chemistry properties via multi-task learning."
Journal of Chemical Information and Modeling, 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Set2Set, global_mean_pool
from torch_geometric.utils import softmax, scatter
import math


# ═══════════════════════════════════════════════════════════════════════
# 1. Atom & Bond Feature Encoders (Champion version)
# ═══════════════════════════════════════════════════════════════════════

class AtomEncoderChampion(nn.Module):
    """Enhanced atom encoder with more chemical features."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        h = hidden_dim
        self.atom_emb = nn.Embedding(100, h // 4)
        self.degree_emb = nn.Embedding(12, h // 8)
        self.charge_emb = nn.Embedding(20, h // 8)
        self.hybrid_emb = nn.Embedding(10, h // 8)
        self.h_emb = nn.Embedding(10, h // 8)
        self.arom_emb = nn.Embedding(2, h // 8)
        self.ring_emb = nn.Embedding(2, h // 8)
        self.chiral_emb = nn.Embedding(4, h // 8)
        self.heavy_emb = nn.Embedding(2, h // 16)
        self.metal_emb = nn.Embedding(2, h // 16)
        self.valence_emb = nn.Embedding(9, h // 16)
        self.cont_proj = nn.Linear(6, h // 4)
        self.out = nn.Linear(h // 4 + 7 * (h // 8) + 3 * (h // 16) + h // 8 + h // 4, hidden_dim)

    def forward(self, x):
        a = self.atom_emb(x[:, 0].long().clamp(0, 99))
        d = self.degree_emb(x[:, 1].long().clamp(0, 11))
        c = self.charge_emb((x[:, 2].long() + 10).clamp(0, 19))
        hy = self.hybrid_emb(x[:, 4].long().clamp(0, 9))
        hh = self.h_emb(x[:, 5].long().clamp(0, 9))
        ar = self.arom_emb(x[:, 3].long().clamp(0, 1))
        ri = self.ring_emb(x[:, 9].long().clamp(0, 1))
        ch = self.chiral_emb(x[:, 10].long().clamp(0, 3))
        he = self.heavy_emb(x[:, 13].long().clamp(0, 1))
        me = self.metal_emb(x[:, 17].long().clamp(0, 1))
        ve = self.valence_emb(x[:, 12].long().clamp(0, 8))
        hs = self.h_emb(x[:, 16].long().clamp(0, 9))
        cn = self.cont_proj(x[:, [6, 7, 8, 11, 14, 15]].float())
        return self.out(torch.cat([a, d, c, hy, hh, ar, ri, ch, he, me, ve, hs, cn], dim=-1))


class BondEncoderChampion(nn.Module):
    """Enhanced bond encoder with RBF distance encoding."""
    def __init__(self, hidden_dim=256, num_rbf=50):
        super().__init__()
        self.type_emb = nn.Embedding(5, hidden_dim // 4)
        self.stereo_emb = nn.Embedding(5, hidden_dim // 8)
        self.arom_emb = nn.Embedding(2, hidden_dim // 8)
        self.feat_proj = nn.Linear(2, hidden_dim // 4)
        self.rbf_centers = nn.Parameter(torch.linspace(0, 10, num_rbf), requires_grad=False)
        self.rbf_gamma = nn.Parameter(torch.tensor(10.0), requires_grad=False)
        self.rbf_proj = nn.Linear(num_rbf, hidden_dim // 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 8 + hidden_dim // 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, edge_attr, edge_dist):
        t = self.type_emb(edge_attr[:, 0].long().clamp(0, 4))
        st = self.stereo_emb(edge_attr[:, 5].long().clamp(0, 4))
        ar = self.arom_emb(edge_attr[:, 6].long().clamp(0, 1))
        f = self.feat_proj(edge_attr[:, [1, 2]].float())
        rbf = torch.exp(-self.rbf_gamma * (edge_dist.unsqueeze(-1) - self.rbf_centers) ** 2)
        d = self.rbf_proj(rbf)
        return self.out(torch.cat([t, st, ar, f, d], dim=-1))


# ═══════════════════════════════════════════════════════════════════════
# 2. Message Passing with LSTM (Champion core)
# ═══════════════════════════════════════════════════════════════════════

class LSTMPassage(MessagePassing):
    """Message passing layer with LSTM update function (champion approach)."""
    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, h_c):
        h, c = h_c
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        h_new, c_new = self.lstm(m, (h, c))
        h_new = self.norm(h + self.dropout(h_new))
        return h_new, (h_new, c_new)

    def message(self, x_i, x_j, edge_attr):
        edge_msg = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return edge_msg


# ═══════════════════════════════════════════════════════════════════════
# 3. Multi-Level Attention Pooling (Champion innovation)
# ═══════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    """Attention-based pooling for node features."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, batch):
        attn_weights = self.attn(x).squeeze(-1)
        attn_weights = softmax(attn_weights, batch)
        out = scatter(x * attn_weights.unsqueeze(-1), batch, dim=0, reduce='sum')
        return out


class MultiLevelAttentionPool(nn.Module):
    """Multi-level attention pooling (champion approach).

    Combines three complementary readout mechanisms:
    1. Attention pooling — learned soft weighting of atom contributions
    2. Set2Set — LSTM-based ordered readout capturing long-range dependencies
    3. Mean pooling — global averaging for overall molecular representation
    """
    def __init__(self, hidden_dim, processing_steps=3):
        super().__init__()
        self.attn_pool = AttentionPooling(hidden_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # attn + Set2Set(2x) + mean
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, batch):
        attn_out = self.attn_pool(x, batch)
        s2s_out = self.set2set(x, batch)
        mean_out = global_mean_pool(x, batch)
        return self.out(torch.cat([attn_out, s2s_out, mean_out], dim=-1))


# ═══════════════════════════════════════════════════════════════════════
# 4. ACSF-inspired Geometric Encoding
# ═══════════════════════════════════════════════════════════════════════

class ACSFEncoding(nn.Module):
    """Atom-Centered Symmetry Functions inspired encoding.

    Encodes pairwise distances via RBF expansion and projects to hidden_dim,
    providing complementary geometric features to the bond encoder.
    """
    def __init__(self, hidden_dim, num_rbf=32):
        super().__init__()
        self.num_rbf = num_rbf
        self.rbf_centers = nn.Parameter(torch.linspace(0, 8, num_rbf), requires_grad=False)
        self.rbf_gamma = nn.Parameter(torch.tensor(8.0), requires_grad=False)
        self.proj = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, pos, edge_index):
        src, dst = edge_index
        dist = (pos[dst] - pos[src]).norm(dim=-1)
        rbf = torch.exp(-self.rbf_gamma * (dist.unsqueeze(-1) - self.rbf_centers) ** 2)
        return self.proj(rbf)


class DirectionEncoding(nn.Module):
    """Encode normalized 3D bond direction vectors.

    Bond direction is critical for dipole moment (mu) prediction,
    as it captures the spatial orientation of charge separation.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, pos, edge_index):
        src, dst = edge_index
        vec = pos[dst] - pos[src]
        norm = vec.norm(dim=-1, keepdim=True) + 1e-8
        return self.proj(vec / norm)


# ═══════════════════════════════════════════════════════════════════════
# 5. Prediction Head (Multi-task)
# ═══════════════════════════════════════════════════════════════════════

class MultiTaskHead(nn.Module):
    """Multi-task prediction head with task-specific layers."""
    def __init__(self, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(out_dim)
        ])

    def forward(self, x):
        shared = self.shared(x)
        outputs = [head(shared) for head in self.task_heads]
        return torch.cat(outputs, dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# 6. Full Champion Model: ape-MPNN
# ═══════════════════════════════════════════════════════════════════════

class MoleculeGNNChampion(nn.Module):
    """ape-MPNN: Attention Pooling Embedded MPNN (Champion architecture).

    Key innovations (Liu et al., JCIM 2021):
    1. LSTM-based message passing (instead of GRU)
    2. Multi-level attention pooling (attention + Set2Set + mean)
    3. ACSF-inspired geometric encoding (radial distribution functions)
    4. Direction encoding for bond orientation (critical for dipole moment)
    5. Multi-task learning head
    """
    def __init__(
        self,
        hidden_dim=256,
        num_layers=5,
        out_dim=12,
        dropout=0.1,
    ):
        super().__init__()
        self.atom_encoder = AtomEncoderChampion(hidden_dim)
        self.bond_encoder = BondEncoderChampion(hidden_dim, num_rbf=50)
        self.acsf = ACSFEncoding(hidden_dim, num_rbf=32)
        self.dir_enc = DirectionEncoding(hidden_dim)

        self.mp_layers = nn.ModuleList([
            LSTMPassage(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.pool = MultiLevelAttentionPool(hidden_dim, processing_steps=3)
        self.head = MultiTaskHead(hidden_dim, out_dim, dropout)

        self.num_layers = num_layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        x = self.atom_encoder(data.x)

        pos = data.pos
        src, dst = data.edge_index
        edge_dist = (pos[src] - pos[dst]).norm(dim=-1)

        edge_attr = (
            self.bond_encoder(data.edge_attr, edge_dist)
            + self.acsf(pos, data.edge_index)
            + self.dir_enc(pos, data.edge_index)
        )

        h = x
        c = torch.zeros_like(x)

        for mp_layer in self.mp_layers:
            h, (h, c) = mp_layer(h, data.edge_index, edge_attr, (h, c))

        x = self.pool(h, data.batch)
        return self.head(x)
