"""
Prepare SMILES + target CSV files for unimol_tools training.
Extracts SMILES from SDF files and combines with mu, R2 targets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_sdf_dirs

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TARGET_COLS = ["mu", "R2"]


def extract_smiles_from_sdf(mol_name, sdf_dirs):
    """Extract SMILES from SDF file."""
    for sdf_dir in sdf_dirs:
        sdf_path = sdf_dir / f"{mol_name}.sdf"
        if sdf_path.exists():
            mol = Chem.MolFromMolFile(str(sdf_path))
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                return smiles
    return None


def prepare_unimol_data(split="train", target_cols=None):
    """Prepare SMILES + target CSV for unimol_tools."""
    if target_cols is None:
        target_cols = ["mu", "R2"]
    
    csv_path = PROCESSED_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)
    
    sdf_dirs = get_sdf_dirs(DATA_DIR)
    print(f"Processing {split} split: {len(df)} molecules")
    print(f"SDF dirs: {[d.name for d in sdf_dirs]}")
    print(f"Target columns: {target_cols}")
    
    smiles_list = []
    valid_count = 0
    invalid_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting SMILES ({split})"):
        mol_name = str(int(row['gdb_idx']))
        smiles = extract_smiles_from_sdf(mol_name, sdf_dirs)
        if smiles is None:
            smiles = ""
            invalid_count += 1
        else:
            valid_count += 1
        smiles_list.append(smiles)
    
    # Create unimol_tools format DataFrame
    target_dict = {f'TARGET_{col}': df[col].values for col in target_cols}
    unimol_df = pd.DataFrame({
        'SMILES': smiles_list,
        **target_dict,
    })
    
    # Filter out empty SMILES
    valid_mask = unimol_df['SMILES'] != ""
    unimol_df = unimol_df[valid_mask].reset_index(drop=True)
    
    # Save to output directory
    out_dir = Path(__file__).parent.parent / "output_unimol"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = out_dir / f"{split}_unimol.csv"
    unimol_df.to_csv(output_path, index=False)
    
    print(f"\n  [{split}] 完成!")
    print(f"    有效分子: {valid_count}")
    print(f"    无效分子: {invalid_count}")
    print(f"    保存到: {output_path}")
    
    return output_path


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        prepare_unimol_data(split)
