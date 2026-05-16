"""
Uni-Mol+ for Molecular Property Prediction

Architecture:
  SMILES -> RDKit 3D conformation -> Uni-Mol+ Transformer -> Property prediction

This implementation adapts the Uni-Mol+ architecture for quantum chemical 
property prediction (mu, R2, etc.)

Run from project root:
  python -m alchemy_mvp.src.unimol.train_unimol
Or directly:
  python alchemy_mvp/src/unimol/train_unimol.py
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, time, platform, logging
from pathlib import Path

# ensure alchemy_mvp package is importable
_PROJECT = Path(__file__).parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

# project imports
from src.utils import get_sdf_dirs, set_seed

# ── Target columns ──
TARGET_COLS = ["mu", "R2"]

# ── Atomic properties ──
ATOMIC_MASS = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    15: 30.974, 16: 32.06, 17: 35.45, 35: 79.904, 53: 126.90,
}

ATOMIC_NUM_MAP = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
}

# ── Configuration ──
CFG = dict(
    data_dir       = _PROJECT.parent / "data",
    out_dir        = _PROJECT / "output_unimol",
    cache_dir      = _PROJECT / "cache_unimol",
    seed           = 42,
    hidden_dim     = 256,
    num_layers     = 6,
    num_heads      = 8,
    ffn_dim        = 1024,
    dropout        = 0.1,
    attention_dropout = 0.1,
    activation_dropout = 0.0,
    max_atoms      = 128,
    num_rbf        = 128,
    cutoff         = 12.0,
    batch_size     = 32,
    epochs         = 200,
    lr             = 1e-4,
    weight_decay   = 1e-4,
    warmup_epochs  = 5,
    patience       = 40,
    lr_factor      = 0.5,
    lr_patience    = 8,
    clip_grad      = 1.0,
)


# ═══════════════════════════════════════════════════════════
# 1. Molecular Conformer Generator
# ═══════════════════════════════════════════════════════════

class ConformerGenerator:
    """Generate 3D conformers from SMILES using RDKit."""
    
    def __init__(self, max_attempts=50, random_seed=42):
        self.max_attempts = max_attempts
        self.random_seed = random_seed
    
    def __call__(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        # Generate conformer
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.maxAttempts = self.max_attempts
        params.useRandomCoords = True
        
        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id == -1:
            # Fallback to random coordinates
            params.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, params)
            if conf_id == -1:
                return None
        
        # Optimize geometry
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            pass
        
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        atomic_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        
        return {
            'coords': coords.astype(np.float32),
            'atomic_nums': atomic_nums.astype(np.int64),
            'smiles': smiles,
        }


# ═══════════════════════════════════════════════════════════
# 2. Radial Basis Functions
# ═══════════════════════════════════════════════════════════

class GaussianRBF(nn.Module):
    """Gaussian Radial Basis Functions for distance encoding."""
    
    def __init__(self, num_rbf=128, cutoff=12.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        centers = torch.linspace(0, cutoff, num_rbf)
        widths = torch.full((num_rbf,), cutoff / num_rbf)
        
        self.register_buffer('centers', centers)
        self.register_buffer('widths', widths)
    
    def forward(self, dist):
        """
        Args:
            dist: (..., 1) distance tensor
        Returns:
            rbf: (..., num_rbf) RBF encoding
        """
        dist = dist.unsqueeze(-1)  # (..., 1, 1)
        rbf = torch.exp(-((dist - self.centers) ** 2) / (2 * self.widths ** 2))
        return rbf


class BesselRBF(nn.Module):
    """Bessel Radial Basis Functions with polynomial envelope."""
    
    def __init__(self, num_rbf=128, cutoff=12.0):
        super().__init__()
        self.cutoff = cutoff
        
        n = torch.arange(1, num_rbf + 1, dtype=torch.float32)
        self.register_buffer("n", torch.pi * n.unsqueeze(0))
        self.register_buffer("norm", torch.tensor((2.0 / cutoff) ** 0.5, dtype=torch.float32))
    
    def forward(self, dist):
        dist = dist.clamp_min(1e-8)
        rbf = self.norm * torch.sin(self.n * dist.unsqueeze(-1) / self.cutoff) / dist.unsqueeze(-1)
        
        p = dist / self.cutoff
        envelope = 1.0 - 6.0 * p**5 + 15.0 * p**4 - 10.0 * p**3
        
        return rbf * envelope.clamp(0, 1).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════
# 3. Uni-Mol+ Components
# ═══════════════════════════════════════════════════════════

class AtomEmbedding(nn.Module):
    """Embed atomic numbers into continuous vectors."""
    
    def __init__(self, num_atoms=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_atoms, hidden_dim)
    
    def forward(self, atomic_nums):
        return self.embedding(atomic_nums)


class DistanceEmbedding(nn.Module):
    """Embed pairwise distances into edge representations."""
    
    def __init__(self, num_rbf=128, hidden_dim=256):
        super().__init__()
        self.rbf = GaussianRBF(num_rbf, cutoff=12.0)
        self.proj = nn.Linear(num_rbf, hidden_dim)
    
    def forward(self, dist_matrix):
        rbf = self.rbf(dist_matrix)  # (N, N, num_rbf)
        return self.proj(rbf)  # (N, N, hidden_dim)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with 3D bias."""
    
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.edge_proj = nn.Linear(hidden_dim, num_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_bias=None, mask=None):
        """
        Args:
            x: (N, B, hidden_dim) atom representations
            edge_bias: (N, N, B, num_heads) edge attention bias
            mask: (N, B) padding mask (True = valid)
        """
        N, B, _ = x.shape
        
        q = self.q_proj(x).view(N, B, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(N, B, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(N, B, self.num_heads, self.head_dim).transpose(0, 1)
        
        # (B, num_heads, N, head_dim)
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, N, N)
        
        if edge_bias is not None:
            # edge_bias: (N, N, B, num_heads) -> (B, num_heads, N, N)
            attn = attn + edge_bias.permute(2, 3, 0, 1)
        
        if mask is not None:
            # mask: (N, B) -> (B, 1, 1, N)
            attn_mask = ~mask.transpose(0, 1).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)
        out = out.transpose(0, 1).contiguous().view(N, B, self.hidden_dim)
        out = self.out_proj(out)
        
        return out


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with 3D bias."""
    
    def __init__(self, hidden_dim=256, num_heads=8, ffn_dim=1024, 
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.0):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, attention_dropout)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
    
    def forward(self, x, edge_bias=None, mask=None):
        # Self-attention with residual
        residual = x
        x = self.self_attn(x, edge_bias, mask)
        x = self.self_attn_dropout(x)
        x = self.self_attn_norm(residual + x)
        
        # FFN with residual
        residual = x
        x = self.activation_dropout(F.gelu(self.fc1(x)))
        x = self.fc2(x)
        x = self.final_dropout(x)
        x = self.final_norm(residual + x)
        
        return x


# ═══════════════════════════════════════════════════════════
# 4. Uni-Mol+ Model
# ═══════════════════════════════════════════════════════════

class UniMolPlus(nn.Module):
    """
    Uni-Mol+ model for molecular property prediction.
    
    Architecture:
      SMILES -> 3D conformer -> atom embeddings + distance bias
           -> Transformer encoder with 3D bias
           -> CLS token pooling -> property prediction
    """
    
    def __init__(self,
                 hidden_dim=256,
                 num_layers=6,
                 num_heads=8,
                 ffn_dim=1024,
                 num_rbf=128,
                 dropout=0.1,
                 attention_dropout=0.1,
                 activation_dropout=0.0,
                 num_targets=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Atom embedding
        self.atom_embed = AtomEmbedding(num_atoms=128, hidden_dim=hidden_dim)
        
        # Distance embedding for edge bias
        self.dist_embed = DistanceEmbedding(num_rbf, hidden_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim, num_heads, ffn_dim,
                dropout, attention_dropout, activation_dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Property prediction head
        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self, atomic_nums, coords, mask=None):
        """
        Args:
            atomic_nums: (N, B) atomic numbers
            coords: (N, B, 3) 3D coordinates
            mask: (N, B) padding mask (True = valid)
        Returns:
            predictions: (B, num_targets)
        """
        N, B = atomic_nums.shape
        
        # Compute pairwise distances
        # coords: (N, B, 3) -> dist_matrix: (N, N, B)
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, B, 3)
        dist_matrix = diff.norm(dim=-1)  # (N, N, B)
        
        # Edge bias from distances
        edge_bias = self.dist_embed(dist_matrix)  # (N, N, B, hidden_dim)
        
        # Atom embeddings
        x = self.atom_embed(atomic_nums)  # (N, B, hidden_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(-1, B, -1)  # (1, B, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=0)  # (N+1, B, hidden_dim)
        
        # Extend mask for CLS token
        if mask is not None:
            cls_mask = torch.ones(1, B, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=0)
        
        # Extend dist_matrix for CLS token: (N, N, B) -> (N+1, N+1, B)
        # Add row for CLS token (distances from CLS to all atoms = 0)
        cls_row = torch.zeros(1, N, B, device=dist_matrix.device)
        dist_matrix = torch.cat([cls_row, dist_matrix], dim=0)  # (N+1, N, B)
        # Add column for CLS token
        cls_col = torch.zeros(N + 1, 1, B, device=dist_matrix.device)
        dist_matrix = torch.cat([dist_matrix, cls_col], dim=1)  # (N+1, N+1, B)
        
        # Extend edge_bias similarly: (N, N, B, hidden_dim) -> (N+1, N+1, B, hidden_dim)
        cls_row_bias = torch.zeros(1, N, B, self.hidden_dim, device=edge_bias.device)
        edge_bias = torch.cat([cls_row_bias, edge_bias], dim=0)  # (N+1, N, B, hidden_dim)
        cls_col_bias = torch.zeros(N + 1, 1, B, self.hidden_dim, device=edge_bias.device)
        edge_bias = torch.cat([edge_bias, cls_col_bias], dim=1)  # (N+1, N+1, B, hidden_dim)
        
        # Transformer encoder
        for layer in self.layers:
            x = layer(x, edge_bias, mask)
        
        x = self.norm(x)
        
        # CLS token output
        cls_out = x[0]  # (B, hidden_dim)
        
        # Property prediction
        predictions = self.property_head(cls_out)  # (B, num_targets)
        
        return predictions


# ═══════════════════════════════════════════════════════════
# 5. Dataset
# ═══════════════════════════════════════════════════════════

class UniMolDataset(torch.utils.data.Dataset):
    """Dataset for Uni-Mol+ molecular property prediction."""
    
    def __init__(self, csv_path, sdf_dirs, scaler=None, is_train=True, max_atoms=128):
        self.df = pd.read_csv(csv_path)
        self.sdf_dirs = sdf_dirs
        self.scaler = scaler
        self.is_train = is_train
        self.max_atoms = max_atoms
        self.conformer_gen = ConformerGenerator()
        
        # Cache for conformers
        self.conformer_cache = {}
    
    def __len__(self):
        return len(self.df)
    
    def _get_conformer_data_from_sdf(self, idx):
        """Load conformer data directly from SDF file."""
        row = self.df.iloc[idx]
        mol_name = row['mol_name'] if 'mol_name' in row else row.iloc[0]
        mol_name = str(mol_name)
        
        # Try to find SDF file
        sdf_path = None
        for sdf_dir in self.sdf_dirs:
            candidate = sdf_dir / f"{mol_name}.sdf"
            if candidate.exists():
                sdf_path = candidate
                break
        
        # Try searching for any SDF file containing mol_name
        if sdf_path is None:
            for sdf_dir in self.sdf_dirs:
                for sdf_file in sdf_dir.glob("*.sdf"):
                    if mol_name in sdf_file.stem:
                        sdf_path = sdf_file
                        break
                if sdf_path:
                    break
        
        if sdf_path and sdf_path.exists():
            mol = Chem.MolFromMolFile(str(sdf_path))
            if mol is not None:
                conf = mol.GetConformer()
                if conf is not None:
                    coords = conf.GetPositions().astype(np.float32)
                    atomic_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
                    return {
                        'coords': coords,
                        'atomic_nums': atomic_nums,
                    }
        
        return None
    
    def _get_smiles(self, idx):
        """Extract SMILES from SDF file or use molecule name."""
        row = self.df.iloc[idx]
        mol_name = row['mol_name'] if 'mol_name' in row else row.iloc[0]
        mol_name = str(mol_name)
        
        # Try to find SDF and extract SMILES
        sdf_path = None
        for sdf_dir in self.sdf_dirs:
            candidate = sdf_dir / f"{mol_name}.sdf"
            if candidate.exists():
                sdf_path = candidate
                break
        
        # Try searching for any SDF file containing mol_name
        if sdf_path is None:
            for sdf_dir in self.sdf_dirs:
                for sdf_file in sdf_dir.glob("*.sdf"):
                    if mol_name in sdf_file.stem:
                        sdf_path = sdf_file
                        break
                if sdf_path:
                    break
        
        if sdf_path and sdf_path.exists():
            mol = Chem.MolFromMolFile(str(sdf_path))
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if smiles:
                    return smiles
        
        # Fallback: return None (will use dummy data)
        return None
    
    def __getitem__(self, idx):
        if idx in self.conformer_cache:
            conformer_data = self.conformer_cache[idx]
        else:
            # Try to load conformer data directly from SDF first
            conformer_data = self._get_conformer_data_from_sdf(idx)
            
            # If SDF loading fails, try SMILES-based conformer generation
            if conformer_data is None:
                smiles = self._get_smiles(idx)
                if smiles:
                    conformer_data = self.conformer_gen(smiles)
            
            # Cache the result
            if conformer_data is not None:
                self.conformer_cache[idx] = conformer_data
        
        if conformer_data is None:
            # Return dummy data
            return self._get_dummy_item(idx)
        
        coords = conformer_data['coords']
        atomic_nums = conformer_data['atomic_nums']
        
        # Truncate to max_atoms
        if len(atomic_nums) > self.max_atoms:
            coords = coords[:self.max_atoms]
            atomic_nums = atomic_nums[:self.max_atoms]
        
        # Get targets
        targets = self.df.iloc[idx][TARGET_COLS].values.astype(np.float32)
        
        if self.scaler is not None:
            targets = self.scaler.transform(targets.reshape(1, -1)).flatten()
        
        return {
            'coords': coords,
            'atomic_nums': atomic_nums,
            'targets': targets,
            'n_atoms': len(atomic_nums),
        }
    
    def _get_dummy_item(self, idx):
        """Return dummy data when conformer generation fails."""
        n_atoms = 10
        return {
            'coords': np.random.randn(n_atoms, 3).astype(np.float32),
            'atomic_nums': np.full(n_atoms, 6, dtype=np.int64),
            'targets': np.zeros(len(TARGET_COLS), dtype=np.float32),
            'n_atoms': n_atoms,
        }


def collate_fn(batch):
    """Collate function for batching molecules."""
    max_n = max(item['n_atoms'] for item in batch)
    batch_size = len(batch)
    
    coords_batch = torch.zeros(max_n, batch_size, 3)
    atomic_nums_batch = torch.zeros(max_n, batch_size, dtype=torch.long)
    mask = torch.zeros(max_n, batch_size, dtype=torch.bool)
    targets_batch = torch.zeros(batch_size, len(TARGET_COLS))
    
    for i, item in enumerate(batch):
        n = item['n_atoms']
        coords_batch[:n, i] = torch.from_numpy(item['coords'])
        atomic_nums_batch[:n, i] = torch.from_numpy(item['atomic_nums'])
        mask[:n, i] = True
        targets_batch[i] = torch.from_numpy(item['targets'])
    
    return {
        'coords': coords_batch,
        'atomic_nums': atomic_nums_batch,
        'mask': mask,
        'targets': targets_batch,
    }


# ═══════════════════════════════════════════════════════════
# 6. Training utilities
# ═══════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, epoch, warmup_scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    
    pbar = tqdm(loader, desc=f"  Epoch {epoch} [Train]", leave=False, ncols=100)
    for batch in pbar:
        coords = batch['coords'].to(device)
        atomic_nums = batch['atomic_nums'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        predictions = model(atomic_nums, coords, mask)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
        optimizer.step()
        
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        
        total_loss += loss.item() * len(targets)
        n += len(targets)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/n:.4f}'
        })
    
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, scaler=None):
    model.eval()
    total_loss, n = 0.0, 0
    preds_list, truths_list = [], []
    
    for batch in tqdm(loader, desc="  Evaluating", leave=False, ncols=100):
        coords = batch['coords'].to(device)
        atomic_nums = batch['atomic_nums'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        
        predictions = model(atomic_nums, coords, mask)
        total_loss += F.mse_loss(predictions, targets, reduction='sum').item()
        n += len(targets)
        
        preds_list.append(predictions.cpu())
        truths_list.append(targets.cpu())
    
    preds = torch.cat(preds_list, dim=0).numpy()
    truths = torch.cat(truths_list, dim=0).numpy()
    
    mae_per = np.mean(np.abs(preds - truths), axis=0)
    rmse_per = np.sqrt(np.mean((preds - truths) ** 2, axis=0))
    r2_per = np.array([
        1 - np.sum((truths[:, i] - preds[:, i]) ** 2)
           / (np.sum((truths[:, i] - truths[:, i].mean()) ** 2) + 1e-8)
        for i in range(preds.shape[1])
    ])
    
    res = {
        "mse": total_loss / n,
        "mae_per": mae_per,
        "rmse_per": rmse_per,
        "r2_per": r2_per,
    }
    
    if scaler is not None:
        preds_raw = scaler.inverse_transform(preds)
        truths_raw = scaler.inverse_transform(truths)
        res["mae_raw"] = np.mean(np.abs(preds_raw - truths_raw), axis=0)
        res["rmse_raw"] = np.sqrt(np.mean((preds_raw - truths_raw) ** 2, axis=0))
        res["r2_raw"] = np.array([
            1 - np.sum((truths_raw[:, i] - preds_raw[:, i]) ** 2)
               / (np.sum((truths_raw[:, i] - truths_raw[:, i].mean()) ** 2) + 1e-8)
            for i in range(preds_raw.shape[1])
        ])
        res["preds_raw"] = preds_raw
        res["truths_raw"] = truths_raw
    
    return res


# ═══════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════

def main():
    cfg = CFG
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    cfg["cache_dir"].mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    
    # Setup logging
    log_file = cfg["out_dir"] / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ]
    )
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    header = "=" * 60
    print(header)
    print("  Uni-Mol+ for Molecular Property Prediction")
    print("  Transformer with 3D distance bias")
    print("=" * 60)
    print(f"  device     : {device}")
    print(f"  data dir   : {cfg['data_dir']}")
    print(f"  cache dir  : {cfg['cache_dir']}")
    
    logger.info(header)
    logger.info("Uni-Mol+ Training Log")
    logger.info(header)
    logger.info(f"device: {device}")
    logger.info(f"data dir: {cfg['data_dir']}")
    logger.info(f"hidden_dim: {cfg['hidden_dim']}, num_layers: {cfg['num_layers']}")
    logger.info(f"num_heads: {cfg['num_heads']}, ffn_dim: {cfg['ffn_dim']}")
    logger.info(f"batch_size: {cfg['batch_size']}, epochs: {cfg['epochs']}, lr: {cfg['lr']}")
    logger.info("")
    
    # ── 1. SDF dirs ──
    sdf_dirs = get_sdf_dirs(cfg["data_dir"])
    print(f"  SDF dirs   : {[d.name for d in sdf_dirs]}")
    
    processed = cfg["data_dir"] / "processed"
    train_csv = processed / "train.csv"
    valid_csv = processed / "valid.csv"
    test_csv  = processed / "test.csv"
    
    # ── 2. Scaler ──
    train_df = pd.read_csv(train_csv)
    scaler = StandardScaler()
    scaler.fit(train_df[TARGET_COLS].values.astype(np.float32))
    print(f"  scaler mu  : mean={scaler.mean_[0]:.4f}  std={scaler.scale_[0]:.4f}")
    print(f"  scaler R2  : mean={scaler.mean_[1]:.4f}  std={scaler.scale_[1]:.4f}")
    
    # ── 3. Datasets ──
    print("\n  Loading datasets ...")
    train_ds = UniMolDataset(train_csv, sdf_dirs, scaler=scaler, is_train=True,
                             max_atoms=cfg['max_atoms'])
    valid_ds = UniMolDataset(valid_csv, sdf_dirs, scaler=scaler, is_train=False,
                             max_atoms=cfg['max_atoms'])
    test_ds  = UniMolDataset(test_csv,  sdf_dirs, scaler=scaler, is_train=False,
                             max_atoms=cfg['max_atoms'])
    print(f"  Train: {len(train_ds)}  |  Valid: {len(valid_ds)}  |  Test: {len(test_ds)}")
    
    # ── 4. DataLoaders ──
    nw = 0 if platform.system() == "Windows" else 2
    pm = device.type == "cuda" and platform.system() != "Windows"
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=pm, collate_fn=collate_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=nw, pin_memory=pm, collate_fn=collate_fn
    )
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=nw, pin_memory=pm, collate_fn=collate_fn
    )
    
    # ── 5. Model ──
    model = UniMolPlus(
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        num_rbf=cfg["num_rbf"],
        dropout=cfg["dropout"],
        attention_dropout=cfg["attention_dropout"],
        activation_dropout=cfg["activation_dropout"],
        num_targets=len(TARGET_COLS),
    ).to(device)
    print(f"\n  Model params : {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=cfg["warmup_epochs"] * len(train_loader))
    plateau = ReduceLROnPlateau(optimizer, mode='min', factor=cfg["lr_factor"],
                                patience=cfg["lr_patience"], min_lr=1e-6)
    
    # ── 6. Train ──
    best_val_mse = float("inf")
    best_state = None
    patience_cnt = 0
    best_epoch = 0
    
    logger.info(f"epoch,train_loss,val_mse,mu_MAE,R2_MAE,mu_R2,R2_R2,lr,improved")
    
    print("\n  Training ...")
    t0 = time.time()
    
    for epoch in range(1, cfg["epochs"] + 1):
        in_warmup = epoch <= cfg["warmup_epochs"]
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch,
                                 warmup_scheduler=warmup if in_warmup else None)
        val_res = evaluate(model, valid_loader, device, scaler=None)
        val_mse = val_res["mse"]
        
        if not in_warmup:
            plateau.step(val_mse)
        lr_now = optimizer.param_groups[0]["lr"]
        
        improved = val_mse < best_val_mse - 1e-7
        if improved:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
        
        mark = " *" if improved else ""
        print(f"  ep {epoch:3d}/{cfg['epochs']} | train {train_loss:.6f} | "
              f"val_mse {val_mse:.6f} | "
              f"mu_MAE {val_res['mae_per'][0]:.4f} | R2_MAE {val_res['mae_per'][1]:.4f} | "
              f"mu_R2 {val_res['r2_per'][0]:.4f} | R2_R2 {val_res['r2_per'][1]:.4f} | "
              f"lr {lr_now:.2e}{mark}")
        
        logger.info(f"{epoch},{train_loss:.6f},{val_mse:.6f},"
                    f"{val_res['mae_per'][0]:.4f},{val_res['mae_per'][1]:.4f},"
                    f"{val_res['r2_per'][0]:.4f},{val_res['r2_per'][1]:.4f},"
                    f"{lr_now:.2e},{improved}")
        
        if patience_cnt >= cfg["patience"]:
            print(f"\n  Early stop @ {epoch} (best: {best_epoch})")
            logger.info(f"Early stop @ epoch {epoch} (best epoch: {best_epoch})")
            break
        if lr_now < 2e-6:
            print(f"\n  LR floor @ {epoch}")
            logger.info(f"Learning rate floor reached @ epoch {epoch}")
            break
    
    train_time = time.time() - t0
    print(f"\n  Trained {train_time:.0f}s  |  best epoch: {best_epoch}")
    logger.info(f"\nTraining completed in {train_time:.0f}s, best epoch: {best_epoch}")
    
    # ── 7. Final eval ──
    model.load_state_dict(best_state)
    
    valid_res = evaluate(model, valid_loader, device, scaler=scaler)
    test_res  = evaluate(model, test_loader,  device, scaler=scaler)
    
    print("\n" + "=" * 60)
    print("  Final Results (original scale)")
    print("=" * 60)
    logger.info("\n" + "=" * 60)
    logger.info("Final Results (original scale)")
    logger.info("=" * 60)
    
    for tag, res in [("VALID", valid_res), ("TEST", test_res)]:
        print(f"  {tag}:")
        logger.info(f"{tag}:")
        for i, col in enumerate(TARGET_COLS):
            print(f"    {col:>4s}  RMSE {res['rmse_raw'][i]:10.4f}  "
                  f"MAE {res['mae_raw'][i]:10.4f}  R^2 {res['r2_raw'][i]:.6f}")
            logger.info(f"  {col}: RMSE={res['rmse_raw'][i]:.4f}, MAE={res['mae_raw'][i]:.4f}, R2={res['r2_raw'][i]:.6f}")
    
    # baseline
    print("\n  Baseline (predict mean on test):")
    logger.info("\nBaseline (predict mean on test):")
    for i, col in enumerate(TARGET_COLS):
        y_test = test_ds.df[col].values.astype(np.float32)
        bl_rmse = np.sqrt(np.mean((y_test - y_test.mean()) ** 2))
        bl_mae  = np.mean(np.abs(y_test - y_test.mean()))
        imp = (1 - test_res["rmse_raw"][i] / bl_rmse) * 100
        print(f"    {col:>4s}  RMSE {bl_rmse:10.4f}  MAE {bl_mae:10.4f}  "
              f"-> model improves {imp:+.1f}%")
        logger.info(f"  {col}: baseline RMSE={bl_rmse:.4f}, MAE={bl_mae:.4f}, improvement={imp:+.1f}%")
    
    # ── 8. Save ──
    torch.save({
        "model_state": best_state,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
        "valid": {
            "mu_r2": float(valid_res["r2_raw"][0]),
            "R2_r2": float(valid_res["r2_raw"][1]),
            "mu_rmse": float(valid_res["rmse_raw"][0]),
            "R2_rmse": float(valid_res["rmse_raw"][1]),
        },
        "test": {
            "mu_r2": float(test_res["r2_raw"][0]),
            "R2_r2": float(test_res["r2_raw"][1]),
            "mu_rmse": float(test_res["rmse_raw"][0]),
            "R2_rmse": float(test_res["rmse_raw"][1]),
        },
    }, cfg["out_dir"] / "model.pt")
    
    print(f"\n  Saved -> {cfg['out_dir'] / 'model.pt'}")
    return model


if __name__ == "__main__":
    main()
