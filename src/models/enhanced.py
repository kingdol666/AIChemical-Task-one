"""
Enhanced MPNN with Transformer-based message passing.
Designed for large-scale training (~162k molecules).

Key improvements over ape-MPNN:
1. Transformer message passing replaces LSTM — multi-head attention + FFN + residual
2. Hidden dim: 512 (up from 256), Layers: 8 (up from 5)
3. Stochastic depth for regularization at scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Set2Set, global_mean_pool
from torch_geometric.utils import softmax, scatter
import math


# ═══════════════════════════════════════════════════════════════════════
# 1. Atom & Bond Feature Encoders (scaled to larger hidden_dim)
# ═══════════════════════════════════════════════════════════════════════

class AtomEncoderEnhanced(nn.Module):
    """Atom encoder for Enhanced model (hidden_dim=512)."""
    def __init__(self, hidden_dim=512):
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


class BondEncoderEnhanced(nn.Module):
    """Bond encoder for Enhanced model (hidden_dim=512)."""
    def __init__(self, hidden_dim=512, num_rbf=64):
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
# 2. Transformer Message Passing (replaces LSTM)
# ═══════════════════════════════════════════════════════════════════════

class TransformerMessagePassing(MessagePassing):
    """Transformer-style message passing layer.

    Edge-conditioned messages + gated attention + FFN + residual + LayerNorm.
    Replaces the LSTM update with a Transformer block that processes
    aggregated neighbor messages through multi-head attention and
    feed-forward networks, enabling parallel computation and better
    gradient flow for large-scale training.
    """
    def __init__(self, hidden_dim, edge_dim, num_heads=8, dropout=0.1):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Edge-conditioned message MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gate: learn how much neighbor info to admit
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Multi-head QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Feed-forward network (Transformer style)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        residual = x

        # 1. Aggregate neighbor messages
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # 2. Gated combination of self and neighbor info
        gate = self.gate(torch.cat([x, m], dim=-1))
        combined = gate * m + (1 - gate) * x

        # 3. Multi-head self-attention (over features, per-node)
        q = self.q_proj(combined).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(combined).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(combined).view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(0, 1).contiguous().view(-1, self.hidden_dim)

        # 4. Residual + LayerNorm
        x = self.norm1(residual + attn_out)

        # 5. Feed-forward + Residual + LayerNorm
        x = self.norm2(x + self.ffn(x))

        return x

    def message(self, x_i, x_j, edge_attr):
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


# ═══════════════════════════════════════════════════════════════════════
# 3. Multi-Level Attention Pooling (scaled)
# ═══════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
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
        return scatter(x * attn_weights.unsqueeze(-1), batch, dim=0, reduce='sum')


class MultiLevelAttentionPool(nn.Module):
    """Multi-level pooling: attention + Set2Set + mean."""
    def __init__(self, hidden_dim, processing_steps=3):
        super().__init__()
        self.attn_pool = AttentionPooling(hidden_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, batch):
        attn_out = self.attn_pool(x, batch)
        s2s_out = self.set2set(x, batch)
        mean_out = global_mean_pool(x, batch)
        return self.out(torch.cat([attn_out, s2s_out, mean_out], dim=-1))


# ═══════════════════════════════════════════════════════════════════════
# 4. Geometric Encodings (scaled)
# ═══════════════════════════════════════════════════════════════════════

class ACSFEncoding(nn.Module):
    """RBF distance encoding for 3D geometry."""
    def __init__(self, hidden_dim, num_rbf=48):
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
    """Normalized bond direction encoding (critical for dipole moment)."""
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
# 5. Multi-Task Head (scaled)
# ═══════════════════════════════════════════════════════════════════════

class MultiTaskHead(nn.Module):
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
# 6. Full Enhanced Model: Transformer MPNN
# ═══════════════════════════════════════════════════════════════════════

class MoleculeGNNEnhanced(nn.Module):
    """Enhanced GNN with Transformer message passing.

    Designed for large-scale training (~162k molecules):
    - Transformer blocks replace LSTM for parallel processing & better gradients
    - Hidden dim 512 (2× champion), Layers 8 (1.6× champion)
    - Stochastic depth for regularization
    - Gated neighbor aggregation
    - Multi-head attention over node features
    """
    def __init__(
        self,
        hidden_dim=512,
        num_layers=8,
        out_dim=12,
        dropout=0.1,
        num_heads=8,
        stochastic_depth_prob=0.1,
    ):
        super().__init__()
        self.atom_encoder = AtomEncoderEnhanced(hidden_dim)
        self.bond_encoder = BondEncoderEnhanced(hidden_dim, num_rbf=64)
        self.acsf = ACSFEncoding(hidden_dim, num_rbf=48)
        self.dir_enc = DirectionEncoding(hidden_dim)

        self.mp_layers = nn.ModuleList([
            TransformerMessagePassing(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.pool = MultiLevelAttentionPool(hidden_dim, processing_steps=3)
        self.head = MultiTaskHead(hidden_dim, out_dim, dropout)

        self.num_layers = num_layers
        self.stochastic_depth_prob = stochastic_depth_prob
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

        for i, mp_layer in enumerate(self.mp_layers):
            # Stochastic depth
            if self.training and self.stochastic_depth_prob > 0:
                if torch.rand(1).item() < self.stochastic_depth_prob * (i / (self.num_layers - 1)):
                    continue
            x = mp_layer(x, data.edge_index, edge_attr)

        x = self.pool(x, data.batch)
        return self.head(x)
