import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.split_data import split_data
from src.train import train
from src.train_ensemble import train_ensemble
from src.predict import predict


def main():
    parser = argparse.ArgumentParser(description="Alchemy MVP - Molecular Property Prediction")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--model_type", type=str, default="champion",
                        choices=["champion", "standard", "enhanced", "mu_r2"],
                        help="Model: champion (ape-MPNN), standard (GNN+Transformer), "
                             "enhanced (Transformer-MPNN), mu_r2 (3D Equivariant GNN)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Enable ensemble training (multiple champion models)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    data_dir = project_root.parent / "data"
    output_dir = project_root / "outputs"
    checkpoint_dir = project_root / "checkpoints"
    scaler_path = output_dir / "target_scaler.pkl"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Alchemy MVP - Molecular Property Prediction")
    print("=" * 60)

    print("\n[STEP 1] Data splitting and standardization...")
    print("-" * 60)
    split_data(data_dir, output_dir, scaler_path)

    print("\n[STEP 2] Training model...")
    print("-" * 60)

    if args.model_type == "mu_r2":
        from src.train_mu_r2 import train_mu_r2
        train_mu_r2(
            data_dir=data_dir,
            output_dir=project_root / "output_gnn_mu_r2",
            cache_dir=project_root / "cache_mu_r2",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
        )
    elif args.ensemble:
        print(f"[INFO] Ensemble mode: training champion models")
        train_ensemble(
            data_dir=data_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
            resume=args.resume,
        )
    else:
        print(f"[INFO] Single model mode: {args.model_type}")
        train(
            data_dir=data_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
            model_type=args.model_type,
            resume=args.resume,
        )

    if args.model_type == "mu_r2":
        print("\n[INFO] mu_r2 model uses its own evaluation — skipping unified predict.")
    else:
        print("\n[STEP 3] Predicting on test set...")
        print("-" * 60)
        predict(
            data_dir=data_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            device_str=args.device,
        )

    print("\n" + "=" * 60)
    print("All steps completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
