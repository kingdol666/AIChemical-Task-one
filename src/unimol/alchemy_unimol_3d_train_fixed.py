#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alchemy Uni-Mol 3D fine-tuning script.

核心改动：
1. 不再把 SDF 转成 SMILES 训练，而是直接从 SDF 读取 atoms + coordinates。
2. 支持 12 维多任务回归：zpve, Cv, gap, G, HOMO, U, alpha, U0, H, LUMO, mu, R2。
3. 可选训练 mu/R2 专家模型，并在最终 answer.csv 中按 valid MAE 自动替换更好的列。
4. 如果 data/processed/train.csv/valid.csv/test.csv 不存在，会从 data/ 下原始 CSV 自动 8:1:1 切分。

运行示例：
python alchemy_unimol_3d_train.py --data_dir data --epochs 100 --batch_size 32 --kfold 5 --train_specialist

依赖：
pip install unimol_tools pandas numpy tqdm scikit-learn joblib huggingface_hub
conda install -c conda-forge rdkit -y
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
except Exception:  # pragma: no cover
    torch = None
from rdkit.Geometry import Point3D

try:
    from rdkit import Chem
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "RDKit 未安装。请先执行：conda install -c conda-forge rdkit -y"
    ) from exc


ALL_TARGETS = [
    "zpve", "Cv", "gap", "G", "HOMO", "U",
    "alpha", "U0", "H", "LUMO", "mu", "R2"
]

MU_R2_TARGETS = ["mu", "R2"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def canonical_col_name(col: str) -> str:
    """把官网 CSV 的长列名规范化成短名。

    例如：
    'zpve\n(Ha, zero point vibrational energy)' -> 'zpve'
    'R2\n(a_0^2, electronic spatial extent)' -> 'R2'
    """
    s = str(col).strip().replace("\r", "\n")
    s = s.split("\n")[0].strip()
    s = re.sub(r"\s*\(.*?\)\s*", "", s).strip()
    return s


def canonicalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: canonical_col_name(c) for c in df.columns}
    df = df.rename(columns=rename).copy()

    # 兼容大小写或奇怪空格
    normalized = {c.lower().replace(" ", ""): c for c in df.columns}
    fixes = {}
    for target in ["gdb_idx", "atom number", *ALL_TARGETS]:
        key = target.lower().replace(" ", "")
        if target not in df.columns and key in normalized:
            fixes[normalized[key]] = target
    if fixes:
        df = df.rename(columns=fixes)

    if "gdb_idx" not in df.columns:
        raise ValueError(f"CSV 中找不到 gdb_idx 列，当前列名为：{list(df.columns)}")

    missing = [c for c in ALL_TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 中缺少目标列：{missing}\n当前列名为：{list(df.columns)}")

    # gdb_idx 转 int，目标转 float
    df["gdb_idx"] = df["gdb_idx"].astype(int)
    for c in ALL_TARGETS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=ALL_TARGETS).reset_index(drop=True)
    if len(df) < before:
        print(f"[WARN] 删除了 {before - len(df)} 行目标值缺失的数据。")
    return df


def find_raw_csv(data_dir: Path) -> Path:
    candidates = []
    for p in data_dir.rglob("*.csv"):
        # 避免把输出文件又当成原始文件
        lower = str(p).lower()
        if any(x in lower for x in ["output", "answer", "prediction", "metric"]):
            continue
        try:
            head = pd.read_csv(p, nrows=5)
            cols = [canonical_col_name(c) for c in head.columns]
            if "gdb_idx" in cols and all(t in cols for t in ALL_TARGETS):
                candidates.append(p)
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"在 {data_dir} 下没有找到包含 gdb_idx 和 12 个目标列的 CSV。"
        )

    # 优先选择非 processed 中的 CSV；如果没有，就用第一个
    candidates = sorted(candidates, key=lambda x: ("processed" in str(x).lower(), len(str(x))))
    return candidates[0]


def split_if_needed(data_dir: Path, seed: int = 42) -> Path:
    processed = data_dir / "processed"
    train_csv = processed / "train.csv"
    valid_csv = processed / "valid.csv"
    test_csv = processed / "test.csv"

    if train_csv.exists() and valid_csv.exists() and test_csv.exists():
        print(f"[DATA] 使用已存在的 processed 数据：{processed}")
        for p in [train_csv, valid_csv, test_csv]:
            df = canonicalize_dataframe(pd.read_csv(p))
            df.to_csv(p, index=False)
        return processed

    print("[DATA] 未发现完整的 data/processed/train.csv/valid.csv/test.csv，开始从原始 CSV 切分。")
    raw_csv = find_raw_csv(data_dir)
    print(f"[DATA] 原始 CSV: {raw_csv}")

    df = canonicalize_dataframe(pd.read_csv(raw_csv))
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    train_df = df.iloc[:n_train].copy()
    valid_df = df.iloc[n_train:n_train + n_valid].copy()
    test_df = df.iloc[n_train + n_valid:].copy()

    processed.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"[DATA] 保存切分结果：")
    print(f"       train: {len(train_df)} -> {train_csv}")
    print(f"       valid: {len(valid_df)} -> {valid_csv}")
    print(f"       test : {len(test_df)} -> {test_csv}")
    return processed


def build_sdf_index(data_dir: Path) -> Dict[int, Path]:
    """递归索引所有 SDF 文件，按文件名中的数字匹配 gdb_idx。"""
    sdf_index: Dict[int, Path] = {}
    sdf_files = list(data_dir.rglob("*.sdf"))
    print(f"[SDF] 在 {data_dir} 下找到 {len(sdf_files)} 个 SDF 文件。")

    for p in sdf_files:
        # 优先使用完整 stem 转 int；否则提取 stem 中最后一组数字
        gid: Optional[int] = None
        try:
            gid = int(p.stem)
        except ValueError:
            nums = re.findall(r"\d+", p.stem)
            if nums:
                gid = int(nums[-1])
        if gid is not None and gid not in sdf_index:
            sdf_index[gid] = p

    if not sdf_index:
        raise FileNotFoundError(f"没有建立任何 SDF 索引，请检查 {data_dir} 下是否存在 .sdf 文件。")

    print(f"[SDF] 成功建立 {len(sdf_index)} 个 gdb_idx -> sdf_path 索引。")
    return sdf_index


def read_sdf_atoms_coords(sdf_path: Path) -> Tuple[List[str], np.ndarray]:
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=True)
    if mol is None:
        # 有些 SDF sanitize 失败，尝试不 sanitize 读取
        mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit 无法读取 SDF: {sdf_path}")

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if not atoms:
        raise ValueError(f"SDF 中没有原子: {sdf_path}")

    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([float(pos.x), float(pos.y), float(pos.z)])
        coords_arr = np.asarray(coords, dtype=np.float64)
    else:
        # 理论上 Alchemy 的 SDF 有 3D 坐标；这里兜底，避免程序中断
        coords_arr = np.zeros((len(atoms), 3), dtype=np.float64)

    if coords_arr.shape != (len(atoms), 3):
        raise ValueError(f"坐标维度异常: {sdf_path}, atoms={len(atoms)}, coords={coords_arr.shape}")

    return atoms, coords_arr


def make_unimol_custom_data(
    df: pd.DataFrame,
    sdf_index: Dict[int, Path],
    targets: Optional[Sequence[str]],
    cache_path: Optional[Path] = None,
) -> Tuple[dict, pd.DataFrame]:
    """把 DataFrame + SDF 转成 Uni-Mol custom coordinate data。

    返回：
    custom_data = {
        'atoms': List[List[str]],
        'coordinates': List[np.ndarray],
        'target': np.ndarray,  # 仅当 targets 不为空
    }
    filtered_df：剔除 SDF 缺失/坏分子之后的 df。
    """
    if cache_path is not None and cache_path.exists():
        print(f"[CACHE] 读取缓存：{cache_path}")
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        return obj["data"], obj["df"]

    atoms_list: List[List[str]] = []
    coords_list: List[np.ndarray] = []
    keep_rows = []
    missing = 0
    bad = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="[SDF] 读取 atoms/coordinates"):
        gid = int(row["gdb_idx"])
        sdf_path = sdf_index.get(gid)
        if sdf_path is None:
            missing += 1
            continue
        try:
            atoms, coords = read_sdf_atoms_coords(sdf_path)
        except Exception as exc:
            bad += 1
            if bad <= 5:
                print(f"[WARN] 跳过坏 SDF: gdb_idx={gid}, path={sdf_path}, err={exc}")
            continue
        atoms_list.append(atoms)
        coords_list.append(coords)
        keep_rows.append(row)

    filtered_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    custom_data = {
        "atoms": atoms_list,
        "coordinates": coords_list,
    }
    if targets:
        y = filtered_df[list(targets)].values.astype(np.float32)
        custom_data["target"] = y

    print(f"[SDF] 有效分子: {len(filtered_df)} / {len(df)} | 缺失 SDF: {missing} | 坏 SDF: {bad}")
    if len(filtered_df) == 0:
        raise RuntimeError("没有任何有效分子，请检查 gdb_idx 与 SDF 文件名是否匹配。")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"data": custom_data, "df": filtered_df}, f)
        print(f"[CACHE] 保存缓存：{cache_path}")

    return custom_data, filtered_df


def ensure_pred_array(predictions, n_targets: int) -> np.ndarray:
    """兼容不同 unimol_tools 版本的预测返回格式。"""
    if isinstance(predictions, pd.DataFrame):
        arr = predictions.values
    elif isinstance(predictions, dict):
        # 优先按 TARGET_0... 或字典顺序拼接
        vals = []
        for k in predictions.keys():
            vals.append(np.asarray(predictions[k]).reshape(-1))
        arr = np.vstack(vals).T
    else:
        arr = np.asarray(predictions)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != n_targets and arr.shape[0] == n_targets:
        arr = arr.T
    if arr.shape[1] != n_targets:
        raise ValueError(f"预测结果维度不对：got {arr.shape}, expected second dim={n_targets}")
    return arr.astype(float)


def patch_unimol_v2_coordinate_bug() -> None:
    """Patch Uni-Mol Tools UniMolV2 raw coordinate loader.

    Some unimol_tools versions call RDKit Conformer.SetAtomPosition(i, coord)
    directly, where coord may be a numpy row/list. On Windows/RDKit this can raise:
    ValueError: cannot extract desired type from sequence.

    This patch converts every coordinate explicitly to rdkit.Geometry.Point3D.
    It is safe for UniMolV2 custom dict input: {'atoms': ..., 'coordinates': ...}.
    """
    try:
        import unimol_tools.data.conformer as conformer
    except Exception:
        return

    def create_mol_from_atoms_and_coords_safe(atoms, coordinates):
        mol = Chem.RWMol()
        atoms_clean = []
        for atom in atoms:
            if isinstance(atom, (int, np.integer)):
                atom_obj = Chem.Atom(int(atom))
                atoms_clean.append(int(atom))
            else:
                atom_obj = Chem.Atom(str(atom))
                atoms_clean.append(str(atom))
            mol.AddAtom(atom_obj)

        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coordinates must have shape [num_atoms, 3], got {coords.shape}")
        if coords.shape[0] != len(atoms_clean):
            raise ValueError(f"atoms/coordinates length mismatch: atoms={len(atoms_clean)}, coords={coords.shape}")

        conf = Chem.Conformer(len(atoms_clean))
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf)

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        return mol.GetMol()

    conformer.create_mol_from_atoms_and_coords = create_mol_from_atoms_and_coords_safe


def train_unimol(
    train_data: dict,
    model_dir: Path,
    targets: Sequence[str],
    args: argparse.Namespace,
) -> None:
    patch_unimol_v2_coordinate_bug()
    from unimol_tools import MolTrain

    model_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"[TRAIN] 开始训练 Uni-Mol 3D 模型 | targets={list(targets)}")
    print(f"[TRAIN] save_path={model_dir}")
    print("=" * 80)

    # 注意：这里不传 patience，而是传官方参数 early_stopping。
    clf = MolTrain(
        task="multilabel_regression" if len(targets) > 1 else "regression",
        data_type="molecule",
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        metrics="mae",
        split="random",
        kfold=args.kfold,
        save_path=str(model_dir),
        remove_hs=False,
        target_normalize="standard",
        max_norm=5.0,
        use_cuda=args.use_cuda,
        use_amp=not args.no_amp,
        use_ddp=False,
        use_gpu=args.use_gpu,
        model_name=args.model_name,
        model_size=args.model_size,
    )
    clf.fit(train_data)
    print(f"[TRAIN] 训练完成：{model_dir}")

    with open(model_dir / "target_names.json", "w", encoding="utf-8") as f:
        json.dump(list(targets), f, ensure_ascii=False, indent=2)


def predict_unimol(model_dir: Path, data: dict, targets: Sequence[str]) -> np.ndarray:
    patch_unimol_v2_coordinate_bug()
    from unimol_tools import MolPredict

    predictor = MolPredict(load_model=str(model_dir))
    # 预测时不需要 target，只传 atoms + coordinates，避免某些版本误处理标签
    pred_input = {
        "atoms": data["atoms"],
        "coordinates": data["coordinates"],
    }
    predictions = predictor.predict(data=pred_input)
    return ensure_pred_array(predictions, len(targets))


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: Sequence[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(targets):
        true = y_true[:, i].astype(float)
        pred = y_pred[:, i].astype(float)
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        rows.append({"target": t, "mae": mae, "rmse": rmse, "r2": r2})
    rows.append({
        "target": "OVERALL_MEAN",
        "mae": float(np.mean([r["mae"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "r2": float(np.mean([r["r2"] for r in rows])),
    })
    return pd.DataFrame(rows)


def evaluate_and_save(
    model_dir: Path,
    split_name: str,
    data: dict,
    df: pd.DataFrame,
    targets: Sequence[str],
    output_dir: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    print(f"\n[EVAL] 预测 {split_name}: model={model_dir.name}, targets={list(targets)}")
    pred = predict_unimol(model_dir, data, targets)
    y_true = df[list(targets)].values.astype(float)

    metrics = calc_metrics(y_true, pred, targets)
    print(metrics.to_string(index=False))

    pred_df = pd.DataFrame(pred, columns=list(targets))
    pred_df.insert(0, "gdb_idx", df["gdb_idx"].astype(int).values)
    pred_df = pred_df.sort_values("gdb_idx").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"pred_{model_dir.name}_{split_name}.csv"
    metric_path = output_dir / f"metrics_{model_dir.name}_{split_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    metrics.to_csv(metric_path, index=False)
    print(f"[SAVE] predictions -> {pred_path}")
    print(f"[SAVE] metrics     -> {metric_path}")
    return pred, metrics


def make_final_answer(
    test_df: pd.DataFrame,
    all_pred: np.ndarray,
    spec_pred: Optional[np.ndarray],
    valid_all_metrics: Optional[pd.DataFrame],
    valid_spec_metrics: Optional[pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    answer = pd.DataFrame(all_pred, columns=ALL_TARGETS)
    answer.insert(0, "gdb_idx", test_df["gdb_idx"].astype(int).values)

    replace_cols = []
    if spec_pred is not None and valid_all_metrics is not None and valid_spec_metrics is not None:
        spec_cols = MU_R2_TARGETS
        for j, col in enumerate(spec_cols):
            all_mae = float(valid_all_metrics.loc[valid_all_metrics["target"] == col, "mae"].iloc[0])
            spec_mae = float(valid_spec_metrics.loc[valid_spec_metrics["target"] == col, "mae"].iloc[0])
            if spec_mae < all_mae:
                answer[col] = spec_pred[:, j]
                replace_cols.append((col, all_mae, spec_mae))

    answer = answer[["gdb_idx", *ALL_TARGETS]].sort_values("gdb_idx").reset_index(drop=True)

    if answer.isna().sum().sum() > 0:
        raise ValueError("answer.csv 中存在 NaN，请检查预测结果。")
    if answer.shape[1] != 13:
        raise ValueError(f"answer.csv 列数错误：{answer.shape[1]}，应为 13。")

    output_dir.mkdir(parents=True, exist_ok=True)
    answer_path = output_dir / "answer.csv"
    answer.to_csv(answer_path, index=False)

    print("\n" + "=" * 80)
    print(f"[ANSWER] 保存最终预测：{answer_path}")
    if replace_cols:
        print("[ANSWER] mu/R2 专家模型替换列：")
        for col, all_mae, spec_mae in replace_cols:
            print(f"         {col}: all_model_MAE={all_mae:.6g} -> specialist_MAE={spec_mae:.6g}")
    else:
        print("[ANSWER] 没有使用专家模型替换列；原因可能是未训练专家模型，或 valid MAE 没有更优。")
    print("=" * 80)

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Alchemy Uni-Mol 3D multi-target regression")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录，默认 data")
    parser.add_argument("--output_dir", type=str, default="output_unimol_3d", help="输出目录")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--early_stopping", type=int, default=20)
    parser.add_argument("--kfold", type=int, default=5, help="Uni-Mol 内部 KFold，追求速度可设为 1")
    parser.add_argument("--model_name", type=str, default="unimolv2", choices=["unimolv1", "unimolv2"])
    parser.add_argument("--model_size", type=str, default="84m", help="unimolv2 可用：84m/164m/310m/570m/1.1B")
    parser.add_argument("--use_gpu", type=str, default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_specialist", action="store_true", help="额外训练 mu/R2 专家模型")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，直接加载已有模型评估/预测")
    parser.add_argument("--rebuild_cache", action="store_true", help="重新解析 SDF，不使用缓存")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "cache"
    model_all_dir = output_dir / "model_all_12"
    model_spec_dir = output_dir / "model_mu_r2"
    pred_dir = output_dir / "predictions"

    args.use_cuda = False if args.cpu else bool(torch is not None and torch.cuda.is_available())
    print("=" * 80)
    print("Alchemy Uni-Mol 3D 训练脚本")
    print(f"data_dir    : {data_dir}")
    print(f"output_dir  : {output_dir}")
    print(f"use_cuda    : {args.use_cuda}")
    print(f"model       : {args.model_name} {args.model_size}")
    print(f"epochs      : {args.epochs}")
    print(f"batch_size  : {args.batch_size}")
    print(f"lr          : {args.lr}")
    print(f"kfold       : {args.kfold}")
    print("=" * 80)

    processed = split_if_needed(data_dir, seed=args.seed)

    train_df = canonicalize_dataframe(pd.read_csv(processed / "train.csv"))
    valid_df = canonicalize_dataframe(pd.read_csv(processed / "valid.csv"))
    test_df = canonicalize_dataframe(pd.read_csv(processed / "test.csv"))

    sdf_index = build_sdf_index(data_dir)

    def cache_path(name: str, targets: Sequence[str]) -> Optional[Path]:
        if args.rebuild_cache:
            return None
        tag = "_".join(targets) if targets else "notarget"
        return cache_dir / f"{name}_{tag}.pkl"

    # 12 维主模型数据
    train_all_data, train_all_df = make_unimol_custom_data(
        train_df, sdf_index, ALL_TARGETS, cache_path("train", ALL_TARGETS)
    )
    valid_all_data, valid_all_df = make_unimol_custom_data(
        valid_df, sdf_index, ALL_TARGETS, cache_path("valid", ALL_TARGETS)
    )
    test_all_data, test_all_df = make_unimol_custom_data(
        test_df, sdf_index, ALL_TARGETS, cache_path("test", ALL_TARGETS)
    )

    if not args.skip_train:
        train_unimol(train_all_data, model_all_dir, ALL_TARGETS, args)
    elif not model_all_dir.exists():
        raise FileNotFoundError(f"--skip_train 已开启，但找不到主模型目录：{model_all_dir}")

    all_valid_pred, all_valid_metrics = evaluate_and_save(
        model_all_dir, "valid", valid_all_data, valid_all_df, ALL_TARGETS, pred_dir
    )
    all_test_pred, all_test_metrics = evaluate_and_save(
        model_all_dir, "test", test_all_data, test_all_df, ALL_TARGETS, pred_dir
    )

    spec_valid_pred = None
    spec_test_pred = None
    spec_valid_metrics = None

    if args.train_specialist:
        # mu/R2 专家模型只使用两个目标，保持同一 SDF 输入
        train_spec_data, train_spec_df = make_unimol_custom_data(
            train_df, sdf_index, MU_R2_TARGETS, cache_path("train", MU_R2_TARGETS)
        )
        valid_spec_data, valid_spec_df = make_unimol_custom_data(
            valid_df, sdf_index, MU_R2_TARGETS, cache_path("valid", MU_R2_TARGETS)
        )
        test_spec_data, test_spec_df = make_unimol_custom_data(
            test_df, sdf_index, MU_R2_TARGETS, cache_path("test", MU_R2_TARGETS)
        )

        if not args.skip_train:
            train_unimol(train_spec_data, model_spec_dir, MU_R2_TARGETS, args)
        elif not model_spec_dir.exists():
            raise FileNotFoundError(f"--skip_train 已开启，但找不到专家模型目录：{model_spec_dir}")

        spec_valid_pred, spec_valid_metrics = evaluate_and_save(
            model_spec_dir, "valid", valid_spec_data, valid_spec_df, MU_R2_TARGETS, pred_dir
        )
        spec_test_pred, spec_test_metrics = evaluate_and_save(
            model_spec_dir, "test", test_spec_data, test_spec_df, MU_R2_TARGETS, pred_dir
        )

        # 确保专家模型和主模型的 test 顺序一致，否则按 gdb_idx 对齐
        if not np.array_equal(test_spec_df["gdb_idx"].values, test_all_df["gdb_idx"].values):
            tmp = pd.DataFrame(spec_test_pred, columns=MU_R2_TARGETS)
            tmp.insert(0, "gdb_idx", test_spec_df["gdb_idx"].values)
            tmp = tmp.set_index("gdb_idx").loc[test_all_df["gdb_idx"].values]
            spec_test_pred = tmp[MU_R2_TARGETS].values

    make_final_answer(
        test_all_df,
        all_test_pred,
        spec_test_pred,
        all_valid_metrics,
        spec_valid_metrics,
        output_dir,
    )

    print("\n[DONE] 全部完成。重点查看：")
    print(f"       {output_dir / 'answer.csv'}")
    print(f"       {pred_dir}")
    print(f"       {model_all_dir}")
    if args.train_specialist:
        print(f"       {model_spec_dir}")


if __name__ == "__main__":
    main()
