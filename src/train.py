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
import logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import MoleculeDataset, TARGET_COLS
from src.models import MoleculeGNN, MoleculeGNNChampion, MoleculeGNNEnhanced
from src.utils import get_device, set_seed, get_sdf_dirs, log_training

NODE_IN_DIM = 18
EDGE_IN_DIM = 7
HIDDEN_DIM = 256
NUM_LAYERS = 5
OUT_DIM = 12
DROPOUT = 0.1
STOCHASTIC_DEPTH = 0.05
RANDOM_SEED = 42
WARMUP_EPOCHS = 5


class PerTargetWeightedLoss(nn.Module):
    """Weighted SmoothL1Loss using scaled-space per-target std.

    For standardized targets (std≈1), we use 1/std weighting to balance
    residual learning difficulty across targets.
    Individual targets can be further boosted via target_boost dict.
    """
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


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, warmup_scheduler=None):
    model.train()
    total_loss = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d} [Train]", leave=True, ncols=120)

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
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

    # Per-target MAE
    mae_per_target = np.mean(np.abs(preds - targets), axis=0)
    mae_scaled = float(np.mean(mae_per_target))

    # Per-target R²
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


@log_training(output_dir="logs", target_names=TARGET_COLS)
def train(
    data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 5e-4,
    device_str: str = "auto",
    model_type: str = "champion",
    resume: bool = False,
    tracker=None,
):
    set_seed(RANDOM_SEED)
    device = get_device(device_str)

    # --- Setup logging ---
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # File logger
    log_file = log_dir / f"train_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(f"train_{model_type}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    # --- CPU optimization ---
    torch.set_num_threads(os.cpu_count() or 4)
    logger.info(f"Using device: {device}")
    logger.info(f"CPU threads: {torch.get_num_threads()}")
    logger.info(f"Model type: {model_type}")

    processed_dir = data_dir / "processed"
    train_csv = processed_dir / "train.csv"
    valid_csv = processed_dir / "valid.csv"

    if not train_csv.exists() or not valid_csv.exists():
        raise FileNotFoundError("Please run split_data.py first to generate train/valid/test splits.")

    scaler_path = output_dir / "target_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"Loaded scaler from {scaler_path}")

    sdf_dirs = get_sdf_dirs(data_dir)
    logger.info(f"Found SDF directories: {[str(d) for d in sdf_dirs]}")

    logger.info("Loading train dataset...")
    train_dataset = MoleculeDataset(train_csv, sdf_dirs, scaler=scaler, is_train=True)
    logger.info(f"Train dataset size: {len(train_dataset)}")

    logger.info("Loading valid dataset...")
    valid_dataset = MoleculeDataset(valid_csv, sdf_dirs, scaler=scaler, is_train=False)
    logger.info(f"Valid dataset size: {len(valid_dataset)}")

    # Compute per-target std in SCALED space for loss weighting.
    # Using scaler.transform on the raw CSV avoids iterating the full dataset.
    train_y_raw = train_dataset.df[TARGET_COLS].values.astype(np.float32)
    train_y_scaled = scaler.transform(train_y_raw)
    scaled_std = train_y_scaled.std(axis=0)  # should all be ~1
    logger.info(f"Scaled-space per-target std: min={scaled_std.min():.4f}, "
          f"max={scaled_std.max():.4f}, mean={scaled_std.mean():.4f}")

    # Windows DataLoader multi-processing causes hang; use num_workers=0
    import platform
    if platform.system() == "Windows":
        num_workers = 0
        pin_memory = False
        logger.info("Windows detected: using num_workers=0 for stability")
    else:
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    if model_type == "champion":
        model = MoleculeGNNChampion(
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            out_dim=OUT_DIM,
            dropout=DROPOUT,
        ).to(device)
        logger.info("Using Champion model (ape-MPNN)")
    elif model_type == "enhanced":
        model = MoleculeGNNEnhanced(
            hidden_dim=512,
            num_layers=8,
            out_dim=OUT_DIM,
            dropout=0.1,
        ).to(device)
        logger.info("Using Enhanced model (Transformer-MPNN, 512/8L)")
    else:
        model = MoleculeGNN(
            node_in_dim=NODE_IN_DIM,
            edge_in_dim=EDGE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            out_dim=OUT_DIM,
            dropout=DROPOUT,
            stochastic_depth_prob=STOCHASTIC_DEPTH,
        ).to(device)
        logger.info("Using Standard model (GNN+Transformer)")

    # --- torch.compile is unstable on Windows, disable by default ---
    import platform
    use_compile = platform.system() != "Windows"
    
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile: ENABLED")
        except Exception:
            logger.info("torch.compile: DISABLED")
    else:
        logger.info("torch.compile: DISABLED (Windows)")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    mu_idx = TARGET_COLS.index("mu")
    criterion = PerTargetWeightedLoss(scaled_std, device, target_boost={mu_idx: 3.0})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # LR Strategy:
    #   Phase 1 — Linear warmup (per-step, first 5 epochs)
    #   Phase 2 — ReduceLROnPlateau (per-epoch): gently decay LR ×0.7
    #             every 3 stagnant epochs for smooth convergence
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=WARMUP_EPOCHS * len(train_loader),
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3,
        min_lr=1e-6,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_valid_mae = float("inf")
    best_model_path = checkpoint_dir / f"best_model_{model_type}.pt"
    last_checkpoint_path = checkpoint_dir / f"last_checkpoint_{model_type}.pt"
    patience_counter = 0
    early_stop_patience = 50  # Increased for longer training
    start_epoch = 1

    if resume and last_checkpoint_path.exists():
        logger.info(f"\nResuming training from {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        
        # Verify model type matches
        saved_model_type = checkpoint.get("model_type", None)
        if saved_model_type is not None and saved_model_type != model_type:
            logger.error(f"Model type mismatch!")
            logger.error(f"  Requested: {model_type}")
            logger.error(f"  Checkpoint: {saved_model_type}")
            logger.info(f"Please use --model_type {saved_model_type} to resume this checkpoint")
            logger.info("Starting training from scratch...")
        else:
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

            logger.info(f"Resuming from epoch {start_epoch}, best MAE: {best_valid_mae:.6f}")
            logger.info(f"LR reset to {lr} (user-specified), scheduler re-initialized")
    elif resume:
        logger.warning(f"\nResume requested but no checkpoint found at {last_checkpoint_path}")
        logger.info("Starting training from scratch...")

    logger.info(f"\nStarting training for {epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Learning rate: {lr}")
    logger.info(f"Warmup: {WARMUP_EPOCHS} epochs (LinearLR)")
    logger.info(f"Loss: PerTargetWeighted SmoothL1")
    logger.info(f"Scheduler: Linear warmup + ReduceLROnPlateau (factor=0.7, patience=3)")
    logger.info(f"Early stopping patience: {early_stop_patience}")
    logger.info("=" * 100)

    total_train_start = time.time()
    tracker_ref = tracker

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        in_warmup = epoch <= WARMUP_EPOCHS
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            warmup_scheduler=warmup_scheduler if in_warmup else None,
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
                    "model_type": model_type,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": plateau_scheduler.state_dict(),
                    "valid_mae_scaled": valid_mae_scaled,
                    "valid_mae_real": valid_mae_real,
                    "best_valid_mae": best_valid_mae,
                    "mae_per_target": mae_per_target.tolist(),
                },
                best_model_path,
            )
            improved = " * BEST"
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": plateau_scheduler.state_dict(),
                "best_valid_mae": best_valid_mae,
                "valid_mae_scaled": valid_mae_scaled,
                "valid_mae_real": valid_mae_real,
            },
            last_checkpoint_path,
        )

        # Tracker logging (for auto CSV + plot via @log_training)
        if tracker_ref is not None:
            tracker_ref.log_epoch(
                epoch,
                train_loss=train_loss,
                val_loss=valid_loss,
                val_mae=valid_mae_scaled,
                val_mae_real=valid_mae_real,
                lr=current_lr,
                mae_per_target=mae_per_target,
                r2_per_target=r2_per_target,
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

        # Print per-target MAE and R² every 10 epochs or when best
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

        # Save intermediate training plot every 10 epochs or when improved
        if tracker_ref is not None and (improved or epoch % 10 == 0 or epoch == start_epoch):
            tracker_ref.save_csv()
            saved_plot = tracker_ref.save_checkpoint_plot()
            if saved_plot:
                tqdm.write(f"  [Plot] {saved_plot}")

        if patience_counter >= early_stop_patience:
            logger.info(f"\nEarly stopping at epoch {epoch}")
            break

    total_time = time.time() - total_train_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info("=" * 100)
    logger.info(f"Training complete in {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info(f"Best valid MAE (scaled): {best_valid_mae:.6f}")
    logger.info(f"Best model saved to {best_model_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Tracker CSV + plot saved to logs/")

    # Load best checkpoint and print final per-target MAE
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    if "mae_per_target" in ckpt:
        logger.info("\nBest per-target MAE (scaled):")
        for i, col in enumerate(TARGET_COLS):
            logger.info(f"  {col:>8s}: {ckpt['mae_per_target'][i]:.6f}")

    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root.parent / "data"
    output_dir = project_root / "outputs"
    checkpoint_dir = project_root / "checkpoints"

    train(
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
    )
