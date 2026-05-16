"""
MuR2SpecialistNet: 3D Equivariant GNN for mu (dipole moment) and R^2 (electronic spatial extent).

Architecture:
  z, pos  -->  3D Equivariant GNN (PaiNN-style)
                   |
         node embeddings + graph embedding
            |                    |
   atom-wise charge head    graph MLP + geometry features
            |                    |
     dipole vector norm         R2 head
            |                    |
            mu                  R2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter

ATOMIC_MASS = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    15: 30.974, 16: 32.06, 17: 35.45, 35: 79.904, 53: 126.90,
}


class BesselRBF(nn.Module):
    """Bessel radial basis functions with polynomial envelope cutoff."""
    def __init__(self, num_rbf=64, cutoff=10.0):
        super().__init__()
        self.cutoff = cutoff
        n = torch.arange(1, num_rbf + 1, dtype=torch.float32)
        self.register_buffer("n", torch.pi * n.unsqueeze(0))
        self.register_buffer("norm", torch.tensor((2.0 / cutoff) ** 0.5, dtype=torch.float32))

    def forward(self, d):
        d = d.clamp_min(1e-8)
        rbf = self.norm * torch.sin(self.n * d.unsqueeze(-1) / self.cutoff) / d.unsqueeze(-1)
        p = d / self.cutoff
        envelope = 1.0 - 6.0 * p**5 + 15.0 * p**4 - 10.0 * p**3
        return rbf * envelope.clamp(0, 1).unsqueeze(-1)


class EquivariantLayer(nn.Module):
    """PaiNN-style equivariant message-passing layer."""
    def __init__(self, hidden_dim, num_rbf=64, dropout=0.05):
        super().__init__()
        Fdim = hidden_dim
        self.rbf = BesselRBF(num_rbf=num_rbf, cutoff=10.0)
        self.mlp_msg = nn.Sequential(
            nn.Linear(Fdim * 2 + num_rbf, Fdim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(Fdim, Fdim * 3),
        )
        self.update_s = nn.Sequential(
            nn.Linear(Fdim * 2, Fdim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(Fdim, Fdim),
        )
        self.update_v = nn.Sequential(
            nn.Linear(Fdim, Fdim), nn.SiLU(), nn.Linear(Fdim, Fdim),
        )
        self.norm_s = nn.LayerNorm(Fdim)
        self.norm_v = nn.LayerNorm(Fdim)

    def forward(self, s, v, pos, edge_index):
        src, dst = edge_index
        vec = pos[src] - pos[dst]
        d = vec.norm(dim=-1) + 1e-8
        r_hat = vec / d.unsqueeze(-1)
        rbf = self.rbf(d)
        mlp_in = torch.cat([s[src], s[dst], rbf], dim=-1)
        ds_msg, gate, rbf_w = self.mlp_msg(mlp_in).chunk(3, dim=-1)
        gate = gate.sigmoid()
        dv_msg = (v[src] * gate.unsqueeze(-1) + rbf_w.unsqueeze(-1) * r_hat.unsqueeze(1))
        ds = scatter(ds_msg, dst, dim=0, dim_size=s.size(0), reduce='sum')
        dv = scatter(dv_msg, dst, dim=0, dim_size=s.size(0), reduce='sum')
        s = s + self.update_s(torch.cat([s, ds], dim=-1))
        v = v + self.update_v(dv.transpose(1, 2)).transpose(1, 2)
        s = self.norm_s(s)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8) \
            * self.norm_v.weight.unsqueeze(0).unsqueeze(-1)
        return s, v


def _atom_mass(z, device):
    m = torch.full((len(z),), 12.0, device=device)
    for k, v in ATOMIC_MASS.items():
        m[z == k] = v
    return m


def compute_geometric_features(pos, z, batch, n_graphs):
    """Per-molecule geometry statistics -> (G, 8)."""
    mass = _atom_mass(z, pos.device)
    feats = torch.zeros(n_graphs, 8, device=pos.device)
    for b in range(n_graphs):
        mask = batch == b
        p = pos[mask]
        m = mass[mask]
        n = p.size(0)
        if n == 0:
            continue
        com = (p * m.unsqueeze(-1)).sum(0) / (m.sum() + 1e-8)
        r = p - com
        rg = ((m * r.pow(2).sum(-1)).sum() / (m.sum() + 1e-8)).sqrt()
        I = torch.zeros(3, 3, device=pos.device)
        for k in range(n):
            rk = r[k]
            I += m[k] * (torch.eye(3, device=pos.device) * rk.dot(rk) - torch.outer(rk, rk))
        eig = torch.linalg.eigvalsh(I)
        I_mean = eig.mean()
        asp = (1.5 * ((eig[2] - I_mean)**2 + (eig[0] - I_mean)**2)
               / (I_mean**2 + 1e-8)).clamp(0, 10)
        feats[b, 0] = rg
        feats[b, 1] = eig[0]
        feats[b, 2] = eig[1]
        feats[b, 3] = eig[2]
        feats[b, 4] = eig[2] / (eig[0] + 1e-8)
        feats[b, 5] = asp
        feats[b, 6] = float(n)
        feats[b, 7] = r.norm(dim=-1).max()
    return feats


class MuR2SpecialistNet(nn.Module):
    """
    3D Equivariant GNN for mu (dipole moment) and R^2 (electronic spatial extent).

    Backbone : PaiNN-style equivariant layers (scalar + vector channels)
    mu head  : atom-wise charge -> dipole vector -> norm -> mu
    R2 head  : graph embedding + geometric statistics -> MLP -> R2
    """
    def __init__(self, hidden_dim=128, num_layers=4, num_rbf=64, geo_feat_dim=8, dropout=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_emb = nn.Embedding(100, hidden_dim // 2)
        self.feat_proj = nn.Linear(17, hidden_dim // 2)
        self.s_init = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EquivariantLayer(hidden_dim, num_rbf, dropout) for _ in range(num_layers)
        ])
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.graph_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.r2_head = nn.Sequential(
            nn.Linear(hidden_dim + geo_feat_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = data.edge_index
        z = x[:, 0].long()
        other = x[:, 1:].float()
        s = self.s_init(torch.cat([self.z_emb(z.clamp(0, 99)), self.feat_proj(other)], dim=-1))
        v = torch.zeros(s.size(0), self.hidden_dim, 3, device=s.device)
        for layer in self.layers:
            s, v = layer(s, v, pos, edge_index)
        q = self.charge_head(s).squeeze(-1)
        mass = _atom_mass(z, s.device)
        n_graphs = int(batch.max().item() + 1)
        M = scatter(mass, batch, dim=0, dim_size=n_graphs)
        com = scatter(pos * mass.unsqueeze(-1), batch, dim=0, dim_size=n_graphs) / (M.unsqueeze(-1) + 1e-8)
        r_rel = pos - com[batch]
        dipole = scatter(q.unsqueeze(-1) * r_rel, batch, dim=0, dim_size=n_graphs)
        mu = dipole.norm(dim=-1)
        s_graph = global_mean_pool(s, batch)
        s_graph = self.graph_embed(s_graph)
        geo = compute_geometric_features(pos, z, batch, n_graphs)
        geo = (geo - geo.mean(0, keepdim=True)) / (geo.std(0, keepdim=True) + 1e-6)
        r2 = self.r2_head(torch.cat([s_graph, geo], dim=-1)).squeeze(-1)
        return torch.stack([mu, r2], dim=-1)
