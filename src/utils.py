import os
import random
import time
import functools
import logging
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "auto"):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def find_csv_file(data_dir: Path) -> Path:
    csv_files = list(data_dir.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")
    if len(csv_files) > 1:
        print(f"[WARN] Multiple CSV files found: {csv_files}. Using the first one.")
    return csv_files[0]


def find_sdf_file(sdf_dirs: list, gdb_idx: int) -> Path:
    for sdf_dir in sdf_dirs:
        sdf_path = sdf_dir / f"{gdb_idx}.sdf"
        if sdf_path.exists():
            return sdf_path
    raise FileNotFoundError(f"SDF file for gdb_idx={gdb_idx} not found in {sdf_dirs}")


def get_sdf_dirs(data_dir: Path) -> list:
    sdf_dirs = []
    for item in data_dir.iterdir():
        if item.is_dir():
            sdf_files = list(item.glob("*.sdf"))
            if len(sdf_files) > 0:
                sdf_dirs.append(item)
    return sdf_dirs


# ---------------------------------------------------------------------------
#  TrainingTracker + @log_training decorator
# ---------------------------------------------------------------------------

class TrainingTracker:
    """Collects per-epoch metrics, writes CSV, and plots training trends."""

    def __init__(self, output_dir, model_name="model", target_names=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.target_names = list(target_names or [])
        self.records = []
        self._csv_path = self.output_dir / f"training_log_{model_name}.csv"
        self._plot_path = self.output_dir / f"training_plot_{model_name}.png"
        self._start_time = time.time()

    # ---- recording ----

    def log_epoch(self, epoch, *, train_loss=None, val_loss=None, val_mae=None,
                  val_mae_real=None, lr=None, mae_per_target=None,
                  r2_per_target=None, **extra):
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_mae_real": val_mae_real,
            "lr": lr,
            "elapsed_s": round(time.time() - self._start_time, 1),
        }
        if mae_per_target is not None:
            names = self.target_names
            for i, v in enumerate(mae_per_target):
                key = names[i] if i < len(names) else f"t{i}"
                record[f"mae_{key}"] = float(v)
        if r2_per_target is not None:
            names = self.target_names
            for i, v in enumerate(r2_per_target):
                key = names[i] if i < len(names) else f"t{i}"
                record[f"r2_{key}"] = float(v)
        record.update(extra)
        self.records.append(record)

    # ---- persistence ----

    def save_csv(self):
        if not self.records:
            return None
        import pandas as pd
        df = pd.DataFrame(self.records)
        df.to_csv(self._csv_path, index=False)
        print(f"[Tracker] CSV  -> {self._csv_path}")
        return self._csv_path

    def plot(self):
        if len(self.records) < 2:
            print("[Tracker] Need >= 2 epochs to plot, skipping.")
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(self.records)
        epochs = df["epoch"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Training: {self.model_name}  "
            f"(epochs: {len(df)}, elapsed: {df['elapsed_s'].iloc[-1]:.0f}s)",
            fontsize=14, fontweight="bold",
        )

        # -- (0,0) Loss curves --
        ax = axes[0, 0]
        if df["train_loss"].notna().any():
            ax.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4")
        if df["val_loss"].notna().any():
            ax.plot(epochs, df["val_loss"], label="Val Loss", color="#d62728")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # -- (0,1) MAE curves --
        ax = axes[0, 1]
        if df["val_mae"].notna().any():
            ax.plot(epochs, df["val_mae"], label="Val MAE (scaled)", color="#2ca02c")
        if "val_mae_real" in df.columns and df["val_mae_real"].notna().any():
            ax.plot(epochs, df["val_mae_real"], label="Val MAE (real)", color="#ff7f0e")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.set_title("MAE Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # -- (1,0) Learning rate --
        ax = axes[1, 0]
        if df["lr"].notna().any():
            ax.plot(epochs, df["lr"], color="#9467bd", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # -- (1,1) Per-target MAE bar chart --
        ax = axes[1, 1]
        mae_cols = [c for c in df.columns if c.startswith("mae_")]
        if mae_cols:
            best_idx = df["val_mae"].idxmin() if df["val_mae"].notna().any() else df.index[-1]
            best_row = df.loc[best_idx]
            names = [c.replace("mae_", "") for c in mae_cols]
            vals = [best_row[c] for c in mae_cols]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
            ax.barh(names, vals, color=colors)
            ax.set_xlabel("MAE (scaled)")
            ax.set_title(f"Per-target MAE  (best epoch {int(best_row['epoch'])})")
            for j, v in enumerate(vals):
                ax.text(v, j, f" {v:.4f}", va="center", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No per-target MAE data", ha="center", va="center",
                    transform=ax.transAxes)

        plt.tight_layout()
        fig.savefig(self._plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Tracker] Plot -> {self._plot_path}")
        return self._plot_path

    def save_checkpoint_plot(self):
        """Save intermediate training plot (call during training, not just at end)."""
        if len(self.records) < 2:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(self.records)
        epochs = df["epoch"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Training: {self.model_name}  "
            f"(epochs: {len(df)}, elapsed: {df['elapsed_s'].iloc[-1]:.0f}s)",
            fontsize=14, fontweight="bold",
        )

        # -- (0,0) Loss curves --
        ax = axes[0, 0]
        if df["train_loss"].notna().any():
            ax.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", linewidth=1.5)
        if df["val_loss"].notna().any():
            ax.plot(epochs, df["val_loss"], label="Val Loss", color="#d62728", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # -- (0,1) MAE curves --
        ax = axes[0, 1]
        if df["val_mae"].notna().any():
            ax.plot(epochs, df["val_mae"], label="Val MAE (scaled)", color="#2ca02c", linewidth=1.5)
        if "val_mae_real" in df.columns and df["val_mae_real"].notna().any():
            ax.plot(epochs, df["val_mae_real"], label="Val MAE (real)", color="#ff7f0e", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.set_title("MAE Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # -- (1,0) Learning rate --
        ax = axes[1, 0]
        if df["lr"].notna().any():
            ax.plot(epochs, df["lr"], color="#9467bd", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # -- (1,1) Per-target MAE bar chart --
        ax = axes[1, 1]
        mae_cols = [c for c in df.columns if c.startswith("mae_")]
        if mae_cols:
            best_idx = df["val_mae"].idxmin() if df["val_mae"].notna().any() else df.index[-1]
            best_row = df.loc[best_idx]
            names = [c.replace("mae_", "") for c in mae_cols]
            vals = [best_row[c] for c in mae_cols]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
            ax.barh(names, vals, color=colors)
            ax.set_xlabel("MAE (scaled)")
            ax.set_title(f"Per-target MAE  (best epoch {int(best_row['epoch'])})")
            for j, v in enumerate(vals):
                ax.text(v, j, f" {v:.4f}", va="center", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No per-target MAE data", ha="center", va="center",
                    transform=ax.transAxes)

        plt.tight_layout()
        fig.savefig(self._plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return self._plot_path

    def finish(self):
        csv_path = self.save_csv()
        plot_path = self.plot()
        return csv_path, plot_path


def log_training(output_dir=None, model_name=None, target_names=None):
    """Decorator that injects a ``tracker`` kwarg into the training function.

    Usage::

        @log_training(output_dir="logs", target_names=["mu", "alpha", ...])
        def train(..., tracker=None):
            for epoch in range(epochs):
                ...
                tracker.log_epoch(epoch, train_loss=..., val_mae=..., ...)

    After ``train()`` returns (or raises), the decorator automatically calls
    ``tracker.finish()`` to save CSV + plot.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = model_name or fn.__name__
            out = Path(output_dir) if output_dir else Path("./logs")
            out = out / f"{name}_{ts}"
            tracker = TrainingTracker(
                output_dir=out,
                model_name=name,
                target_names=target_names,
            )
            kwargs["tracker"] = tracker
            try:
                result = fn(*args, **kwargs)
            finally:
                tracker.finish()
            return result
        return wrapper
    return decorator
