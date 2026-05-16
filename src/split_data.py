import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import find_csv_file

TARGET_COLS = ["zpve", "Cv", "gap", "G", "HOMO", "U", "alpha", "U0", "H", "LUMO", "mu", "R2"]
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1


def clean_column_name(col: str) -> str:
    return col.split("\n")[0].strip()


def split_data(data_dir: Path, output_dir: Path, scaler_path: Path):
    print("[INFO] Looking for CSV file...")
    csv_path = find_csv_file(data_dir)
    print(f"[INFO] Found CSV: {csv_path}")

    print("[INFO] Reading CSV...")
    df = pd.read_csv(csv_path, header=0)

    df.columns = [clean_column_name(c) for c in df.columns]
    print(f"[INFO] Cleaned columns: {df.columns.tolist()}")

    if "gdb_idx" not in df.columns:
        raise ValueError("Column 'gdb_idx' not found in CSV!")

    for col in TARGET_COLS:
        if col not in df.columns:
            raise ValueError(f"Target column '{col}' not found in CSV!")

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Total samples: {len(df)}")

    df = df[["gdb_idx"] + TARGET_COLS].copy()

    print("[INFO] Splitting data (8:1:1)...")
    df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n = len(df_shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_df = df_shuffled.iloc[:n_train].reset_index(drop=True)
    valid_df = df_shuffled.iloc[n_train:n_train + n_valid].reset_index(drop=True)
    test_df = df_shuffled.iloc[n_train + n_valid:].reset_index(drop=True)

    print(f"[INFO] Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    print("[INFO] Fitting StandardScaler on train set...")
    scaler = StandardScaler()
    scaler.fit(train_df[TARGET_COLS])

    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Scaler saved to {scaler_path}")

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    valid_df.to_csv(processed_dir / "valid.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    print(f"[INFO] Saved train.csv, valid.csv, test.csv to {processed_dir}")
    print("[INFO] Data split complete!")

    return train_df, valid_df, test_df, scaler


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_dir = project_root.parent / "data"
    output_dir = project_root / "outputs"
    scaler_path = output_dir / "target_scaler.pkl"

    split_data(data_dir, output_dir, scaler_path)
