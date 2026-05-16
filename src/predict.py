import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from pathlib import Path
import pickle
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import MoleculeDataset, TARGET_COLS
from src.models import MoleculeGNN, MoleculeGNNChampion, MoleculeGNNEnhanced
from src.utils import get_device, set_seed, get_sdf_dirs

RANDOM_SEED = 42


def detect_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    is_enhanced = any(k.startswith("mp_layers.0.norm1") for k in state_dict)
    is_champion = any(k.startswith("head.task_heads") for k in state_dict)
    is_standard = any(k.startswith("gnn_blocks") for k in state_dict)

    if not is_enhanced and not is_champion and not is_standard:
        raise ValueError(
            f"Cannot detect model type from {ckpt_path}. "
            "Keys match neither Enhanced, Champion nor Standard architecture."
        )

    atom_emb_shape = state_dict["atom_encoder.atom_emb.weight"].shape
    hidden_dim = atom_emb_shape[1] * 4

    if is_enhanced:
        num_layers = sum(
            1 for k in state_dict
            if k.startswith("mp_layers.") and k.endswith(".norm1.weight")
        ) or 8
        num_heads = int(state_dict.get("mp_layers.0.num_heads", 8))
    elif is_champion:
        num_layers = sum(
            1 for k in state_dict
            if k.startswith("mp_layers.") and k.endswith(".lstm.weight_ih")
        ) or 5
        num_heads = None
    else:
        num_layers = sum(
            1 for k in state_dict
            if k.startswith("gnn_blocks.") and k.endswith(".conv.eps")
        ) or 5
        num_heads = None

    return {
        "is_enhanced": is_enhanced,
        "is_champion": is_champion,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "out_dim": 12,
        "dropout": 0.1,
    }


def load_model_from_checkpoint(ckpt_path, device):
    info = detect_model_from_checkpoint(ckpt_path, device)

    if info["is_enhanced"]:
        model = MoleculeGNNEnhanced(
            hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"],
            out_dim=info["out_dim"],
            dropout=info["dropout"],
        ).to(device)
        tag = f"Enhanced (Transformer-MPNN), hidden={info['hidden_dim']}, layers={info['num_layers']}"
    elif info["is_champion"]:
        model = MoleculeGNNChampion(
            hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"],
            out_dim=info["out_dim"],
            dropout=info["dropout"],
        ).to(device)
        tag = f"Champion (ape-MPNN), hidden={info['hidden_dim']}, layers={info['num_layers']}"
    else:
        model = MoleculeGNN(
            node_in_dim=18,
            edge_in_dim=7,
            hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"],
            out_dim=info["out_dim"],
            dropout=info["dropout"],
            stochastic_depth_prob=0.0,
        ).to(device)
        tag = f"Standard (GNN+Transformer), hidden={info['hidden_dim']}, layers={info['num_layers']}"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded {tag} from {ckpt_path.name}")

    return model, ckpt


def find_best_models(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    models = []

    ensemble_info_path = checkpoint_dir / "ensemble_info.pkl"
    if ensemble_info_path.exists():
        with open(ensemble_info_path, "rb") as f:
            info = pickle.load(f)
        for path_str in info.get("model_paths", []):
            p = Path(path_str)
            if p.exists():
                models.append(p)
        if models:
            print(f"[INFO] Found ensemble with {len(models)} models")
            return models

    # Search named checkpoints first (best_model_champion.pt etc.)
    named = sorted(checkpoint_dir.glob("best_model_*.pt"))
    if named:
        return named

    # Fall back to legacy checkpoint name
    best_path = checkpoint_dir / "best_model.pt"
    if best_path.exists():
        return [best_path]

    for subdir in sorted(checkpoint_dir.glob("model_*")):
        for pt_file in sorted(subdir.glob("best_model_*.pt")):
            models.append(pt_file)

    return models


def predict(
    data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    device_str: str = "auto",
):
    set_seed(RANDOM_SEED)
    device = get_device(device_str)
    print(f"[INFO] Using device: {device}")

    processed_dir = data_dir / "processed"
    test_csv = processed_dir / "test.csv"

    if not test_csv.exists():
        raise FileNotFoundError("Please run split_data.py first to generate test split.")

    scaler_path = output_dir / "target_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[INFO] Loaded scaler from {scaler_path}")

    sdf_dirs = get_sdf_dirs(data_dir)
    print(f"[INFO] Found SDF directories: {[str(d) for d in sdf_dirs]}")

    print("[INFO] Loading test dataset...")
    test_dataset = MoleculeDataset(test_csv, sdf_dirs, scaler=scaler, is_train=False)
    print(f"[INFO] Test dataset size: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model_paths = find_best_models(checkpoint_dir)
    if not model_paths:
        raise FileNotFoundError(
            f"No best_model checkpoint found in {checkpoint_dir}. "
            "Please run train.py first."
        )

    all_preds_scaled = []
    model_count = len(model_paths)
    print(f"[INFO] Found {model_count} model(s) for evaluation\n")

    for i, ckpt_path in enumerate(model_paths):
        print(f"[{'Ensemble' if model_count > 1 else 'Single'} model {i+1}/{model_count}: {ckpt_path}")
        model, ckpt = load_model_from_checkpoint(ckpt_path, device)

        model.eval()
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                preds.append(out.cpu().numpy())

        preds_scaled = np.concatenate(preds, axis=0)
        all_preds_scaled.append(preds_scaled)

        if model_count == 1:
            all_targets = []
            all_gdb_idx = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    all_targets.append(batch.y.cpu().numpy())
                    all_gdb_idx.append(batch.gdb_idx.cpu().numpy())
            targets_scaled = np.concatenate(all_targets, axis=0)
            gdb_idx = np.concatenate(all_gdb_idx, axis=0).flatten()

    all_preds_scaled = np.array(all_preds_scaled)
    if model_count > 1:
        preds_scaled = np.mean(all_preds_scaled, axis=0)
        print(f"\n[INFO] Ensemble: averaged predictions from {model_count} models")

        all_targets = []
        all_gdb_idx = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                all_targets.append(batch.y.cpu().numpy())
                all_gdb_idx.append(batch.gdb_idx.cpu().numpy())
        targets_scaled = np.concatenate(all_targets, axis=0)
        gdb_idx = np.concatenate(all_gdb_idx, axis=0).flatten()
    else:
        preds_scaled = all_preds_scaled[0]

    preds_real = scaler.inverse_transform(preds_scaled)

    targets_real = scaler.inverse_transform(targets_scaled)
    mae_scaled = np.mean(np.abs(preds_scaled - targets_scaled), axis=0)
    mae_real = np.mean(np.abs(preds_real - targets_real), axis=0)

    print("\n[INFO] Test MAE (scaled space):")
    for i, col in enumerate(TARGET_COLS):
        print(f"  {col:>8s}: {mae_scaled[i]:.6f}")
    print(f"  {'Overall':>8s}: {np.mean(mae_scaled):.6f}")

    print("\n[INFO] Test MAE (real physical scale):")
    for i, col in enumerate(TARGET_COLS):
        print(f"  {col:>8s}: {mae_real[i]:.6f}")
    print(f"  {'Overall':>8s}: {np.mean(mae_real):.6f}")

    answer_df = pd.DataFrame(preds_real, columns=TARGET_COLS)
    answer_df.insert(0, "gdb_idx", gdb_idx)
    answer_df = answer_df.sort_values("gdb_idx").reset_index(drop=True)

    if answer_df.isna().any().any():
        print("[WARN] NaN values detected in predictions! Filling with 0...")
        answer_df = answer_df.fillna(0)

    answer_path = output_dir / "answer.csv"
    answer_df.to_csv(answer_path, index=False)
    print(f"\n[INFO] Answer saved to {answer_path}")
    print(f"[INFO] Answer shape: {answer_df.shape}")
    print(f"[INFO] Columns: {answer_df.columns.tolist()}")

    return answer_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint file (overrides auto-detection)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root.parent / "data"
    output_dir = project_root / "outputs"
    checkpoint_dir = project_root / "checkpoints"

    predict(
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        device_str=args.device,
    )
