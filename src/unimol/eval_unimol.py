"""Evaluation script for UniMol models."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from unimol_tools import MolPredict

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "output_unimol_all" / "unimol_model_all"
TEST_CSV = PROJECT_ROOT / "output_unimol_all" / "test_unimol.csv"

ALL_TARGETS = ["zpve", "Cv", "gap", "G", "HOMO", "U", "alpha", "U0", "H", "LUMO", "mu", "R2"]
TARGET_COLS = [f"TARGET_{col}" for col in ALL_TARGETS]


def evaluate(model_path=None, test_csv=None):
    model_path = model_path or MODEL_PATH
    test_csv = test_csv or TEST_CSV

    print("=== Loading model and evaluating ===")
    clf = MolPredict(load_model=str(model_path))
    predictions = clf.predict(data=str(test_csv))

    df = pd.read_csv(test_csv)
    true_values = df[TARGET_COLS].values

    if isinstance(predictions, dict):
        pred_values = np.array([predictions[col] for col in TARGET_COLS]).T
    else:
        pred_values = predictions

    print()
    for i, col in enumerate(ALL_TARGETS):
        true = true_values[:, i]
        pred = pred_values[:, i]
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        print(f"{col:>8s}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")


if __name__ == "__main__":
    evaluate()
