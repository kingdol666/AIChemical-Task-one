"""Training script for MuR2SpecialistNet (mu & R2 specialist)."""

import os, sys, time, platform, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import src.dataset as _ds
from src.dataset import MoleculeDataset
from src.models.mu_r2 import MuR2SpecialistNet
from src.utils import get_sdf_dirs, set_seed, log_training

TARGET_COLS = ["mu", "R2"]


def train_epoch(model, loader, optimizer, device, epoch, warmup_scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"  Epoch {epoch} [Train]", leave=False, ncols=100)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(batch), batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/n:.4f}'})
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, scaler=None):
    model.eval()
    total_loss, n = 0.0, 0
    preds_list, truths_list = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        total_loss += F.mse_loss(pred, batch.y, reduction='sum').item()
        n += batch.num_graphs
        preds_list.append(pred.cpu())
        truths_list.append(batch.y.cpu())

    preds = torch.cat(preds_list, dim=0).numpy()
    truths = torch.cat(truths_list, dim=0).numpy()
    mae_per = np.mean(np.abs(preds - truths), axis=0)
    rmse_per = np.sqrt(np.mean((preds - truths) ** 2, axis=0))
    r2_per = np.array([
        1 - np.sum((truths[:, i] - preds[:, i]) ** 2)
           / (np.sum((truths[:, i] - truths[:, i].mean()) ** 2) + 1e-8)
        for i in range(preds.shape[1])
    ])
    res = {"mse": total_loss / n, "mae_per": mae_per, "rmse_per": rmse_per, "r2_per": r2_per}
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
    return res


@log_training(output_dir="logs", target_names=TARGET_COLS)
def train_mu_r2(
    data_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 2e-4,
    device_str: str = "auto",
    patience: int = 50,
    tracker=None,
):
    set_seed(42)
    device = torch.device(device_str if device_str != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "training.log"
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8")])
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("  MuR2SpecialistNet - PaiNN-style 3D Equivariant GNN")
    print("  mu : atom charges -> dipole norm")
    print("  R2 : graph embed + geometry stats -> MLP")
    print("=" * 60)
    print(f"  device: {device}")

    sdf_dirs = get_sdf_dirs(data_dir)
    processed = data_dir / "processed"

    train_df = pd.read_csv(processed / "train.csv")
    scaler = StandardScaler()
    scaler.fit(train_df[TARGET_COLS].values.astype(np.float32))

    train_ds = MoleculeDataset(processed / "train.csv", sdf_dirs, scaler=scaler, is_train=True, cache_dir=cache_dir, target_cols=TARGET_COLS)
    valid_ds = MoleculeDataset(processed / "valid.csv", sdf_dirs, scaler=scaler, is_train=False, cache_dir=cache_dir, target_cols=TARGET_COLS)
    test_ds  = MoleculeDataset(processed / "test.csv",  sdf_dirs, scaler=scaler, is_train=False, cache_dir=cache_dir, target_cols=TARGET_COLS)
    print(f"  Train: {len(train_ds)}  |  Valid: {len(valid_ds)}  |  Test: {len(test_ds)}")

    nw = 0 if platform.system() == "Windows" else 2
    pm = device.type == "cuda" and platform.system() != "Windows"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pm)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False, num_workers=nw, pin_memory=pm)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2, shuffle=False, num_workers=nw, pin_memory=pm)

    model = MuR2SpecialistNet(hidden_dim=128, num_layers=4, num_rbf=64, geo_feat_dim=8, dropout=0.05).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=3 * len(train_loader))
    plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6)

    best_val_mse = float("inf")
    best_state = None
    patience_cnt = 0
    best_epoch = 0

    print("\n  Training ...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        in_warmup = epoch <= 3
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
        print(f"  ep {epoch:3d}/{epochs} | train {train_loss:.6f} | "
              f"val_mse {val_mse:.6f} | "
              f"mu_MAE {val_res['mae_per'][0]:.4f} | R2_MAE {val_res['mae_per'][1]:.4f} | "
              f"lr {lr_now:.2e}{mark}")

        if tracker is not None:
            tracker.log_epoch(
                epoch,
                train_loss=train_loss,
                val_loss=val_mse,
                val_mae=float(np.mean(val_res["mae_per"])),
                lr=lr_now,
                mae_per_target=val_res["mae_per"],
                r2_per_target=val_res["r2_per"],
            )

        if patience_cnt >= patience or lr_now < 2e-6:
            print(f"\n  Early stop @ {epoch} (best: {best_epoch})")
            break

    print(f"\n  Trained {time.time()-t0:.0f}s  |  best epoch: {best_epoch}")

    model.load_state_dict(best_state)
    valid_res = evaluate(model, valid_loader, device, scaler=scaler)
    test_res  = evaluate(model, test_loader,  device, scaler=scaler)

    print("\n" + "=" * 60)
    print("  Final Results (original scale)")
    print("=" * 60)
    for tag, res in [("VALID", valid_res), ("TEST", test_res)]:
        print(f"  {tag}:")
        for i, col in enumerate(TARGET_COLS):
            print(f"    {col:>4s}  RMSE {res['rmse_raw'][i]:10.4f}  "
                  f"MAE {res['mae_raw'][i]:10.4f}  R^2 {res['r2_raw'][i]:.6f}")

    torch.save({
        "model_state": best_state,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }, output_dir / "model.pt")
    print(f"\n  Saved -> {output_dir / 'model.pt'}")
    return model
