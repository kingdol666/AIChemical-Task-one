"""
Alchemy MVP - Champion Model Training with Ensemble
Based on ape-MPNN (NJU_Chem team, Tencent Alchemy Contest 2019)
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from pathlib import Path
import argparse
import sys
import pickle
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import MoleculeDataset, TARGET_COLS
from src.models import MoleculeGNNChampion, MoleculeGNNEnhanced
from src.utils import get_device, set_seed, get_sdf_dirs, log_training

NODE_IN_DIM = 18
EDGE_IN_DIM = 7
HIDDEN_DIM = 256
NUM_LAYERS = 5
OUT_DIM = 12
DROPOUT = 0.1
RANDOM_SEED = 42
WARMUP_EPOCHS = 5


class PerTargetWeightedLoss(nn.Module):
    def __init__(self, scaled_std, device, target_boost=None):
        super().__init__()
        std = torch.tensor(scaled_std, dtype=torch.float32).to(device)
        weights = 1.0 / (std + 1e-6)
        weights = weights / weights.mean()
        if target_boost is not None:
            for idx, factor in target_boost.items():
                weights[idx] *= factor
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        loss_per_dim = F.smooth_l1_loss(pred, target, reduction='none')
        return (loss_per_dim * self.weights).mean()


class ACSFRegularizer(nn.Module):
    """ACSF-inspired regularizer to enforce geometric consistency."""
    def __init__(self, num_rbf=32):
        super().__init__()
        self.num_rbf = num_rbf
        self.rbf_centers = nn.Parameter(torch.linspace(0, 8, num_rbf), requires_grad=False)
        self.rbf_gamma = nn.Parameter(torch.tensor(8.0), requires_grad=False)

    def forward(self, pos, edge_index):
        src, dst = edge_index
        vec = pos[dst] - pos[src]
        dist = vec.norm(dim=-1)
        rbf = torch.exp(-self.rbf_gamma * (dist.unsqueeze(-1) - self.rbf_centers) ** 2)
        return rbf.mean()


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, warmup_scheduler=None, acsf_reg=None, acsf_weight=0.0):
    model.train()
    total_loss = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d} [Train]", leave=True, ncols=120)

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)

        if acsf_reg is not None and acsf_weight > 0:
            acsf_loss = acsf_reg(batch.pos, batch.edge_index)
            loss = loss + acsf_weight * acsf_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples


def evaluate_with_metrics(model, loader, criterion, device, scaler=None, epoch=0):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d} [Valid]", leave=True, ncols=120)

    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            all_preds.append(out.cpu().float())
            all_targets.append(batch.y.cpu().float())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    mae_per_target = np.mean(np.abs(preds - targets), axis=0)
    mae_scaled = float(np.mean(mae_per_target))

    r2_per_target = np.array([
        1 - np.sum((targets[:, i] - preds[:, i]) ** 2) / (np.sum((targets[:, i] - targets[:, i].mean()) ** 2) + 1e-8)
        for i in range(preds.shape[1])
    ])

    mae_real = None
    r2_real = None
    if scaler is not None:
        preds_real = scaler.inverse_transform(preds)
        targets_real = scaler.inverse_transform(targets)
        mae_real_per = np.mean(np.abs(preds_real - targets_real), axis=0)
        mae_real = float(np.mean(mae_real_per))
        r2_real = np.array([
            1 - np.sum((targets_real[:, i] - preds_real[:, i]) ** 2) / (np.sum((targets_real[:, i] - targets_real[:, i].mean()) ** 2) + 1e-8)
            for i in range(preds_real.shape[1])
        ])

    avg_loss = total_loss / total_samples
    return avg_loss, mae_scaled, mae_real, mae_per_target, r2_per_target, r2_real


def create_model(model_type, device):
    if model_type == "champion":
        model = MoleculeGNNChampion(
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            out_dim=OUT_DIM,
            dropout=DROPOUT,
        ).to(device)
    elif model_type == "enhanced":
        model = MoleculeGNNEnhanced(
            hidden_dim=512,
            num_layers=8,
            out_dim=OUT_DIM,
            dropout=0.1,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def train_single_model(
    data_dir, output_dir, checkpoint_dir, model_type, epochs, lr, batch_size, device,
    train_dataset, valid_dataset, scaler, scaled_std, train_loader, valid_loader,
    resume=False, acsf_weight=0.0, load_from=None, tracker=None
):
    set_seed(RANDOM_SEED)
    
    model = create_model(model_type, device)
    
    if load_from is not None and Path(load_from).exists():
        print(f"[INFO] Loading pretrained weights from {load_from}")
        checkpoint = torch.load(load_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    use_compile = False
    
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[INFO] torch.compile: ENABLED")
        except Exception:
            print("[INFO] torch.compile: DISABLED")
    else:
        print("[INFO] torch.compile: DISABLED")

    print(f"[INFO] Model: {model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    from src.dataset import TARGET_COLS as _TARGET_COLS
    mu_idx = _TARGET_COLS.index("mu")
    criterion = PerTargetWeightedLoss(scaled_std, device, target_boost={mu_idx: 3.0})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS * len(train_loader))
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3,
        min_lr=1e-6,
    )

    acsf_reg = ACSFRegularizer(num_rbf=32).to(device) if acsf_weight > 0 else None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_valid_mae = float("inf")
    best_model_path = checkpoint_dir / f"best_model_{model_type}.pt"
    last_checkpoint_path = checkpoint_dir / f"last_checkpoint_{model_type}.pt"
    patience_counter = 0
    early_stop_patience = 25
    start_epoch = 1

    if resume and last_checkpoint_path.exists():
        print(f"\n[INFO] Resuming training from {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_valid_mae = checkpoint.get("best_valid_mae", float("inf"))

        # Override LR with user-specified value, reset scheduler accordingly
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3, min_lr=1e-6,
        )

        print(f"[INFO] Resuming from epoch {start_epoch}, best MAE: {best_valid_mae:.6f}")
        print(f"[INFO] LR reset to {lr} (user-specified), scheduler re-initialized")
    elif resume:
        print(f"\n[WARN] Resume requested but no checkpoint found at {last_checkpoint_path}")
        print("[INFO] Starting training from scratch...")

    print(f"\n[INFO] Starting training for {epochs} epochs...")
    print(f"[INFO] Batch size: {batch_size}, Learning rate: {lr}")
    print(f"[INFO] ACSF weight: {acsf_weight}")
    print(f"[INFO] Warmup: {WARMUP_EPOCHS} epochs (LinearLR)")
    print(f"[INFO] Scheduler: Linear warmup + ReduceLROnPlateau (factor=0.7, patience=3)")
    print(f"[INFO] Early stopping patience: {early_stop_patience}")
    print("=" * 100)

    total_train_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        in_warmup = epoch <= WARMUP_EPOCHS
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            warmup_scheduler=warmup_scheduler if in_warmup else None,
            acsf_reg=acsf_reg, acsf_weight=acsf_weight,
        )
        valid_loss, valid_mae_scaled, valid_mae_real, mae_per_target, r2_per_target, r2_real = evaluate_with_metrics(
            model, valid_loader, criterion, device, scaler, epoch
        )

        if not in_warmup:
            plateau_scheduler.step(valid_mae_scaled)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - total_train_start
        avg_epoch_time = elapsed_total / epoch
        remaining_epochs = epochs - epoch
        eta_str = time.strftime("%H:%M:%S", time.gmtime(avg_epoch_time * remaining_epochs))

        progress_pct = epoch / epochs * 100
        bar_len = 30
        filled = int(bar_len * epoch / epochs)
        bar = "█" * filled + "░" * (bar_len - filled)

        improved = ""
        if valid_mae_scaled < best_valid_mae:
            best_valid_mae = valid_mae_scaled
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": plateau_scheduler.state_dict(),
                    "valid_mae_scaled": valid_mae_scaled,
                    "valid_mae_real": valid_mae_real,
                    "best_valid_mae": best_valid_mae,
                    "mae_per_target": mae_per_target.tolist(),
                    "model_type": model_type,
                },
                best_model_path,
            )
            improved = " * BEST"
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": plateau_scheduler.state_dict(),
                "best_valid_mae": best_valid_mae,
                "valid_mae_scaled": valid_mae_scaled,
                "valid_mae_real": valid_mae_real,
                "model_type": model_type,
            },
            last_checkpoint_path,
        )

        tqdm.write(
            f"[{bar}] {progress_pct:5.1f}% | "
            f"Ep {epoch:03d}/{epochs} | "
            f"Train {train_loss:.6f} | "
            f"Val {valid_loss:.6f} | "
            f"MAE {valid_mae_scaled:.6f} | "
            f"Real {valid_mae_real:.6f} | "
            f"LR {current_lr:.1e} | "
            f"{epoch_time:.1f}s | "
            f"ETA {eta_str}"
            f"{improved}"
        )

        if improved or epoch % 10 == 0:
            per_target_str = "  ".join(
                f"{TARGET_COLS[i][:4]}:{mae_per_target[i]:.4f}"
                for i in range(OUT_DIM)
            )
            tqdm.write(f"  Per-target MAE: {per_target_str}")
            r2_str = "  ".join(
                f"{TARGET_COLS[i][:4]}:{r2_per_target[i]:.3f}"
                for i in range(OUT_DIM)
            )
            tqdm.write(f"  Per-target R²:  {r2_str}")

        if tracker is not None:
            tracker.log_epoch(
                epoch,
                train_loss=train_loss,
                val_loss=valid_loss,
                val_mae=valid_mae_scaled,
                val_mae_real=valid_mae_real,
                lr=current_lr,
                mae_per_target=mae_per_target,
                r2_per_target=r2_per_target if r2_real is not None else r2_per_target,
            )

        # Save intermediate training plot every 10 epochs or when improved
        if tracker is not None and (improved or epoch % 10 == 0 or epoch == start_epoch):
            tracker.save_csv()
            saved_plot = tracker.save_checkpoint_plot()
            if saved_plot:
                tqdm.write(f"  [Plot] {saved_plot}")

        if patience_counter >= early_stop_patience:
            tqdm.write(f"\n[INFO] Early stopping at epoch {epoch}")
            break

    total_time = time.time() - total_train_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print("=" * 100)
    print(f"[INFO] Training complete in {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"[INFO] Best valid MAE (scaled): {best_valid_mae:.6f}")
    print(f"[INFO] Best model saved to {best_model_path}")

    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    if "mae_per_target" in ckpt:
        print("\n[INFO] Best per-target MAE (scaled):")
        for i, col in enumerate(TARGET_COLS):
            print(f"  {col:>8s}: {ckpt['mae_per_target'][i]:.6f}")

    return best_model_path, best_valid_mae


def ensemble_predict(models, data, scaler, device):
    """Uncertainty-weighted ensemble prediction."""
    all_preds = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            pred = model(data).cpu().numpy()
            all_preds.append(pred)
    
    all_preds = np.array(all_preds)
    
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    
    weights = 1.0 / (std_pred + 1e-6)
    weights = weights / weights.sum(axis=0, keepdims=True)
    
    ensemble_pred = np.sum(all_preds * weights, axis=0)
    
    return ensemble_pred


def train_ensemble(
    data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    epochs: int = 120,
    batch_size: int = 128,
    lr: float = 5e-4,
    device_str: str = "auto",
    model_types: list = None,
    resume: bool = False,
    acsf_weight: float = 0.01,
):
    if model_types is None:
        model_types = ["champion"]

    set_seed(RANDOM_SEED)
    device = get_device(device_str)

    torch.set_num_threads(os.cpu_count() or 4)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] CPU threads: {torch.get_num_threads()}")
    print(f"[INFO] Model types: {model_types}")

    processed_dir = data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    valid_csv = processed_dir / "valid.csv"

    if not train_csv.exists() or not valid_csv.exists():
        raise FileNotFoundError("Please run split_data.py first to generate train/valid/test splits.")

    scaler_path = output_dir / "target_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[INFO] Loaded scaler from {scaler_path}")

    sdf_dirs = get_sdf_dirs(data_dir)
    print(f"[INFO] Found SDF directories: {[str(d) for d in sdf_dirs]}")

    print("[INFO] Loading train dataset...")
    train_dataset = MoleculeDataset(train_csv, sdf_dirs, scaler=scaler, is_train=True)
    print(f"[INFO] Train dataset size: {len(train_dataset)}")

    print("[INFO] Loading valid dataset...")
    valid_dataset = MoleculeDataset(valid_csv, sdf_dirs, scaler=scaler, is_train=False)
    print(f"[INFO] Valid dataset size: {len(valid_dataset)}")

    train_y_raw = train_dataset.df[TARGET_COLS].values.astype(np.float32)
    train_y_scaled = scaler.transform(train_y_raw)
    scaled_std = train_y_scaled.std(axis=0)
    print(f"[INFO] Scaled-space per-target std: min={scaled_std.min():.4f}, "
          f"max={scaled_std.max():.4f}, mean={scaled_std.mean():.4f}")

    import platform
    if platform.system() == "Windows":
        num_workers = 0
        pin_memory = False
        print("[INFO] Windows detected: using num_workers=0 for stability")
    else:
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    from src.utils import TrainingTracker
    ensemble_models = []
    ensemble_maes = []

    for i, mtype in enumerate(model_types):
        print(f"\n{'='*100}")
        print(f"[INFO] Training Model {i+1}/{len(model_types)}: {mtype}")
        print(f"{'='*100}")

        model_checkpoint_dir = checkpoint_dir / f"model_{i+1}_{mtype}"
        model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        per_model_tracker = TrainingTracker(
            output_dir=Path("logs") / f"{mtype}_{i+1}_{ts}",
            model_name=f"{mtype}_{i+1}",
            target_names=TARGET_COLS,
        )

        best_model_path, best_mae = train_single_model(
            data_dir=data_dir,
            output_dir=output_dir,
            checkpoint_dir=model_checkpoint_dir,
            model_type=mtype,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            scaler=scaler,
            scaled_std=scaled_std,
            train_loader=train_loader,
            valid_loader=valid_loader,
            resume=resume,
            acsf_weight=acsf_weight,
            tracker=per_model_tracker,
        )
        per_model_tracker.finish()

        ensemble_models.append(best_model_path)
        ensemble_maes.append(best_mae)

    print(f"\n{'='*100}")
    print(f"[INFO] Ensemble Training Complete!")
    print(f"[INFO] Individual model MAEs: {[f'{m:.6f}' for m in ensemble_maes]}")
    print(f"[INFO] Ensemble models saved to: {[str(p) for p in ensemble_models]}")

    ensemble_info = {
        "model_paths": [str(p) for p in ensemble_models],
        "model_maes": ensemble_maes,
        "model_types": model_types,
        "ensemble_mae": np.mean(ensemble_maes),
    }

    ensemble_path = checkpoint_dir / "ensemble_info.pkl"
    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble_info, f)

    print(f"[INFO] Ensemble info saved to {ensemble_path}")

    return ensemble_models, ensemble_maes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_types", type=str, nargs="+", default=["champion"])
    parser.add_argument("--acsf_weight", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root.parent / "data"
    output_dir = project_root / "outputs"
    checkpoint_dir = project_root / "checkpoints"

    train_ensemble(
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        model_types=args.model_types,
        resume=args.resume,
        acsf_weight=args.acsf_weight,
    )
