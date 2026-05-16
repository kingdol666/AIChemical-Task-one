import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, Set2Set, GraphNorm, AttentionalAggregation,
    global_mean_pool, global_add_pool, global_max_pool,
)
from torch_geometric.utils import softmax, scatter
import math


# ═══════════════════════════════════════════════════════════════════════
# 1. 3D Geometric Encoding
# ═══════════════════════════════════════════════════════════════════════

class GaussianRBF(nn.Module):
    """Expand scalar distances with Gaussian RBF kernels."""
    def __init__(self, num_rbf=64, cutoff=10.0):
        super().__init__()
        self.cutoff = cutoff
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_rbf), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(10.0 / cutoff), requires_grad=False)

    def forward(self, dist):
        rbf = torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)
        cutoff_mask = 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1)
        cutoff_mask = cutoff_mask.unsqueeze(-1)
        return rbf * cutoff_mask


class SphericalEncoding(nn.Module):
    """Encode angular information from 3D coordinates."""
    def __init__(self, num_spherical=16, out_dim=64):
        super().__init__()
        self.num_spherical = num_spherical
        self.proj = nn.Linear(num_spherical * 2, out_dim)

    def forward(self, pos, edge_index):
        src, dst = edge_index
        vec = pos[dst] - pos[src]
        norm = vec.norm(dim=-1, keepdim=True) + 1e-8
        vec_norm = vec / norm

        theta = torch.acos(vec_norm[:, 1].clamp(-1, 1))
        phi = torch.atan2(vec_norm[:, 2], vec_norm[:, 0])

        theta_emb = torch.cat([
            torch.sin(theta.unsqueeze(-1) * (i + 1))
            for i in range(self.num_spherical)
        ], dim=-1)
        phi_emb = torch.cat([
            torch.cos(phi.unsqueeze(-1) * (i + 1))
            for i in range(self.num_spherical)
        ], dim=-1)

        return self.proj(torch.cat([theta_emb, phi_emb], dim=-1))


# ═══════════════════════════════════════════════════════════════════════
# 2. Encoders
# ═══════════════════════════════════════════════════════════════════════

class AtomEncoder(nn.Module):
    """Encode 11 atom features → hidden_dim."""
    def __init__(self, hidden_dim=384):
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
        self.cont_proj = nn.Linear(3, h // 8)
        self.out = nn.Linear(h // 4 + 7 * (h // 8) + h // 8, hidden_dim)

    def forward(self, x):
        a = self.atom_emb(x[:, 0].long().clamp(0, 99))
        d = self.degree_emb(x[:, 1].long().clamp(0, 11))
        c = self.charge_emb((x[:, 2].long() + 10).clamp(0, 19))
        hy = self.hybrid_emb(x[:, 4].long().clamp(0, 9))
        hh = self.h_emb(x[:, 5].long().clamp(0, 9))
        ar = self.arom_emb(x[:, 3].long().clamp(0, 1))
        ri = self.ring_emb(x[:, 9].long().clamp(0, 1))
        ch = self.chiral_emb(x[:, 10].long().clamp(0, 3))
        cn = self.cont_proj(x[:, 6:9].float())
        return self.out(torch.cat([a, d, c, hy, hh, ar, ri, ch, cn], dim=-1))


class BondEncoder(nn.Module):
    """Encode bond features + 3D distance RBF + spherical encoding."""
    def __init__(self, hidden_dim=384, num_rbf=64, num_spherical=16):
        super().__init__()
        emb_dim = hidden_dim // 4
        spherical_out = 64
        
        self.type_emb = nn.Embedding(5, emb_dim)
        self.feat_proj = nn.Linear(3, emb_dim)
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=10.0)
        self.rbf_proj = nn.Linear(num_rbf, emb_dim)
        self.spherical = SphericalEncoding(num_spherical=num_spherical)
        
        total_in_dim = emb_dim * 3 + spherical_out
        self.out = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, edge_attr, edge_dist, pos, edge_index):
        t = self.type_emb(edge_attr[:, 0].long().clamp(0, 4))
        f = self.feat_proj(edge_attr[:, 1:4].float())
        d = self.rbf_proj(self.rbf(edge_dist))
        s = self.spherical(pos, edge_index)
        return self.out(torch.cat([t, f, d, s], dim=-1))


# ═══════════════════════════════════════════════════════════════════════
# 3. GNN + Transformer Hybrid Block
# ═══════════════════════════════════════════════════════════════════════

class DistanceGate(nn.Module):
    def __init__(self, edge_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, edge_dim // 2),
            nn.GELU(),
            nn.Linear(edge_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, edge_attr):
        return self.net(edge_attr)


class GNNBlock(nn.Module):
    """GINEConv + distance gate + FFN."""
    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.conv = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=edge_dim,
        )
        self.dist_gate = DistanceGate(edge_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        gate = self.dist_gate(edge_attr)
        _, dst = edge_index
        gate_per_node = scatter(gate.squeeze(-1), dst, dim=0, reduce='mean',
                                dim_size=x.size(0)).unsqueeze(-1)
        h = h * gate_per_node
        h = self.norm1(x + self.dropout(h))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class TransformerBlock(nn.Module):
    """Self-attention on node features for long-range dependencies."""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=False
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        max_nodes = batch.bincount().max().item()
        batch_size = batch.max().item() + 1

        x_list = []
        for b in range(batch_size):
            mask = batch == b
            x_b = x[mask]
            x_list.append(x_b)

        x_padded = torch.zeros(batch_size, max_nodes, x.size(-1), device=x.device)
        mask_padded = torch.ones(batch_size, max_nodes, device=x.device, dtype=torch.bool)
        for b, x_b in enumerate(x_list):
            x_padded[b, :x_b.size(0)] = x_b
            mask_padded[b, :x_b.size(0)] = False

        x_transposed = x_padded.transpose(0, 1)
        h, _ = self.attention(x_transposed, x_transposed, x_transposed, key_padding_mask=mask_padded)
        h = h.transpose(0, 1)

        h_out = torch.zeros_like(x)
        for b in range(batch_size):
            mask = batch == b
            h_out[mask] = h[b, :mask.sum().item()]

        h_out = self.norm1(x + self.dropout(h_out))
        h_out = self.norm2(h_out + self.dropout(self.ffn(h_out)))
        return h_out


# ═══════════════════════════════════════════════════════════════════════
# 4. Virtual Node
# ═══════════════════════════════════════════════════════════════════════

class VirtualNode(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.vn_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x, batch, vn_state):
        vn_feat = scatter(x, batch, dim=0, reduce='mean')
        vn_state = vn_state + self.update(vn_feat).mean(dim=0, keepdim=True)
        return x + self.proj(vn_state).expand_as(x), vn_state


# ═══════════════════════════════════════════════════════════════════════
# 5. Multi-Scale Readout
# ═══════════════════════════════════════════════════════════════════════

class MultiScaleReadout(nn.Module):
    """Combine Set2Set, AttentionalAggregation, and multi-scale pooling."""
    def __init__(self, hidden_dim, processing_steps=4):
        super().__init__()
        self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps)
        self.attn = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            ),
            nn=nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x, batch):
        s2s = self.set2set(x, batch)
        att = self.attn(x, batch)
        return self.out(torch.cat([s2s, att], dim=-1))


# ═══════════════════════════════════════════════════════════════════════
# 6. Prediction Head
# ═══════════════════════════════════════════════════════════════════════

class PredictionHead(nn.Module):
    def __init__(self, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.norm1(self.fc1(x)))
        h = self.dropout(h)
        h = F.gelu(self.norm2(self.fc2(h)))
        h = self.dropout(h)
        return self.fc3(h)


# ═══════════════════════════════════════════════════════════════════════
# 7. Full Model: GNN + Transformer + 3D Geometry
# ═══════════════════════════════════════════════════════════════════════

class MoleculeGNN(nn.Module):
    def __init__(
        self,
        node_in_dim=11,
        edge_in_dim=4,
        hidden_dim=384,
        num_layers=5,
        out_dim=12,
        dropout=0.1,
        stochastic_depth_prob=0.0,
    ):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

        self.gnn_blocks = nn.ModuleList([
            GNNBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=8, dropout=dropout)
            for _ in range(2)
        ])

        self.virtual_node = VirtualNode(hidden_dim)

        self.readout = MultiScaleReadout(hidden_dim, processing_steps=4)
        self.head = PredictionHead(hidden_dim, out_dim, dropout)

        self.stochastic_depth_prob = stochastic_depth_prob
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

        edge_attr = self.bond_encoder(data.edge_attr, edge_dist, pos, data.edge_index)

        vn_state = self.virtual_node.vn_init.clone()

        for i, block in enumerate(self.gnn_blocks):
            if self.training and self.stochastic_depth_prob > 0:
                if torch.rand(1).item() < self.stochastic_depth_prob:
                    continue
            x = block(x, data.edge_index, edge_attr)
            x, vn_state = self.virtual_node(x, data.batch, vn_state)

        for t_block in self.transformer_blocks:
            x = t_block(x, data.batch)

        x = self.readout(x, data.batch)
        return self.head(x)
