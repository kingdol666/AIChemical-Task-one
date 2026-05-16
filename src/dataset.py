import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Dataset, Data
from rdkit import Chem
from rdkit.Chem import rdchem
from tqdm import tqdm
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import find_sdf_file, get_sdf_dirs

TARGET_COLS = ["zpve", "Cv", "gap", "G", "HOMO", "U", "alpha", "U0", "H", "LUMO", "mu", "R2"]

BOND_TYPE_MAP = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
    Chem.BondType.OTHER: 4,
}

HYBRIDIZATION_MAP = {
    rdchem.HybridizationType.S: 0,
    rdchem.HybridizationType.SP: 1,
    rdchem.HybridizationType.SP2: 2,
    rdchem.HybridizationType.SP3: 3,
    rdchem.HybridizationType.SP3D: 4,
    rdchem.HybridizationType.SP3D2: 5,
    rdchem.HybridizationType.SP2D: 6,
    rdchem.HybridizationType.OTHER: 7,
}

ELECTRONEGATIVITY = {
    1: 2.20, 2: 0.0, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
    9: 3.98, 10: 0.0, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
    16: 2.58, 17: 3.16, 18: 0.0, 19: 0.82, 20: 1.00, 26: 1.83, 29: 1.90,
    30: 1.65, 35: 2.96, 53: 2.66,
}

ATOMIC_MASS = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 15: 30.974,
    16: 32.06, 17: 35.45, 35: 79.904, 53: 126.90,
}

ATOMIC_RADII = {
    1: 0.38, 6: 0.77, 7: 0.75, 8: 0.73, 9: 0.71, 15: 1.06,
    16: 1.02, 17: 0.99, 35: 1.14, 53: 1.33,
}

VALENCE_ELECTRONS = {
    1: 1, 6: 4, 7: 5, 8: 6, 9: 7, 15: 5, 16: 6, 17: 7, 35: 7, 53: 7,
}


def get_atom_feature_enhanced(atom):
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    is_aromatic = int(atom.GetIsAromatic())
    hybridization = HYBRIDIZATION_MAP.get(atom.GetHybridization(), 7)
    total_h = atom.GetTotalNumHs()
    electronegativity = ELECTRONEGATIVITY.get(atomic_num, 0.0)
    atomic_mass = ATOMIC_MASS.get(atomic_num, 0.0)
    num_radical_electrons = atom.GetNumRadicalElectrons()
    is_in_ring = int(atom.IsInRing())
    chirality = int(atom.GetChiralTag())
    
    atomic_radius = ATOMIC_RADII.get(atomic_num, 0.0)
    valence_electrons = VALENCE_ELECTRONS.get(atomic_num, 0)
    is_heavy = int(atomic_num > 1)
    implicit_valence = atom.GetValence(Chem.ValenceType.IMPLICIT)
    explicit_valence = atom.GetValence(Chem.ValenceType.EXPLICIT)
    num_hs = atom.GetNumExplicitHs()
    is_metal = int(atomic_num in [3, 4, 11, 12, 13, 19, 20, 26, 29, 30])
    
    return [
        atomic_num, degree, formal_charge, is_aromatic, hybridization,
        total_h, electronegativity, atomic_mass, num_radical_electrons,
        is_in_ring, chirality, atomic_radius, valence_electrons,
        is_heavy, implicit_valence, explicit_valence, num_hs, is_metal,
    ]


def get_bond_feature_enhanced(bond, mol):
    bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 4)
    is_conjugated = int(bond.GetIsConjugated())
    is_in_ring = int(bond.IsInRing())
    
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    conf = mol.GetConformer()
    if conf is not None:
        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        bond_length = ((pos_i.x - pos_j.x)**2 + (pos_i.y - pos_j.y)**2 + (pos_i.z - pos_j.z)**2)**0.5
    else:
        bond_length = 0.0
    
    bond_order = bond.GetBondTypeAsDouble()
    stereo = int(bond.GetStereo())
    is_aromatic_bond = int(bond.GetIsAromatic())
    
    return [
        bond_type, is_conjugated, is_in_ring, bond_length,
        bond_order, stereo, is_aromatic_bond,
    ]


def mol_to_graph_enhanced(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_feature_enhanced(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index_list = []
    edge_attr_list = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_attr = get_bond_feature_enhanced(bond, mol)

        edge_index_list.append([i, j])
        edge_attr_list.append(edge_attr)

        edge_index_list.append([j, i])
        edge_attr_list.append(edge_attr)

    if len(edge_index_list) > 0:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)

    conf = mol.GetConformer()
    if conf is not None:
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
    else:
        pos = torch.zeros((x.size(0), 3), dtype=torch.float)

    return x, edge_index, edge_attr, pos


class MoleculeDataset(Dataset):
    def __init__(self, csv_path, sdf_dirs, scaler=None, is_train=True, cache_dir=None, target_cols=None):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.sdf_dirs = sdf_dirs
        self.scaler = scaler
        self.is_train = is_train
        self.target_cols = target_cols if target_cols is not None else TARGET_COLS
        self.cache_dir = cache_dir or (Path(__file__).parent.parent / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        csv_hash = self._compute_hash(csv_path)
        self.cache_file = self.cache_dir / f"dataset_{csv_hash}.pt"

        self.data_list = []

        if self.cache_file.exists():
            self._load_from_cache()
        else:
            self._load_from_sdf()
            self._save_to_cache()

    def _compute_hash(self, csv_path):
        with open(csv_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:10]

    def _load_from_cache(self):
        total = len(self.df)
        print(f"  Cache found: {self.cache_file.name}")
        cache_size_mb = self.cache_file.stat().st_size / (1024 * 1024)
        print(f"  Loading {total} graphs from cache ({cache_size_mb:.1f} MB) ...")

        pbar = tqdm(total=100, desc="  Reading cache", ncols=80, unit="%")
        self.data_list = torch.load(self.cache_file, weights_only=False)
        pbar.update(100)
        pbar.close()

        print(f"  Loaded {len(self.data_list)} graphs from cache.")

    def _load_from_sdf(self):
        total = len(self.df)
        scaler_mean = self.scaler.mean_ if self.scaler is not None else None
        scaler_scale = self.scaler.scale_ if self.scaler is not None else None

        print(f"  No cache found. Loading {total} molecules from SDF files ...")
        self.data_list = [None] * total

        pbar = tqdm(range(total), desc="  Parsing SDF", ncols=100, unit="mol",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        skipped = 0
        for idx in pbar:
            row = self.df.iloc[idx]
            gdb_idx = int(row["gdb_idx"])

            try:
                sdf_path = find_sdf_file(self.sdf_dirs, gdb_idx)
            except FileNotFoundError:
                skipped += 1
                sdf_path = None

            mol = None
            if sdf_path is not None:
                mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
                if mol is None:
                    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False)

            if mol is None:
                x = torch.zeros((1, 18), dtype=torch.float)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 7), dtype=torch.float)
                pos = torch.zeros((1, 3), dtype=torch.float)
            else:
                x, edge_index, edge_attr, pos = mol_to_graph_enhanced(mol)

            y_values = row[self.target_cols].values.astype(np.float32)
            if scaler_mean is not None:
                y_scaled = (y_values - scaler_mean) / scaler_scale
                y = torch.tensor(y_scaled, dtype=torch.float).unsqueeze(0)
            else:
                y = torch.tensor(y_values, dtype=torch.float).unsqueeze(0)

            self.data_list[idx] = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos,
                y=y, gdb_idx=torch.tensor([gdb_idx], dtype=torch.long),
            )

        pbar.close()
        print(f"  Parsed {total} molecules.")
        if skipped > 0:
            print(f"  [WARN] {skipped} molecules missing SDF files — replaced with dummy graphs.")

    def _save_to_cache(self):
        print(f"  Saving cache to {self.cache_file.name} ...", end=" ", flush=True)
        torch.save(self.data_list, self.cache_file)

        size_mb = self.cache_file.stat().st_size / (1024 * 1024)
        print(f"Done ({size_mb:.1f} MB)")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
