"""
使用 unimol_tools 对所有12维目标进行多任务回归训练

目标维度:
  zpve, Cv, gap, G, HOMO, U, alpha, U0, H, LUMO, mu, R2

运行方式:
  python alchemy_mvp/src/unimol/train_all_targets.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

# 确保 alchemy_mvp 包可导入
_PROJECT = Path(__file__).parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem

# ── 配置 ──
DATA_DIR = _PROJECT.parent / "data"
OUT_DIR = _PROJECT / "output_unimol_all"
PROCESSED_DIR = DATA_DIR / "processed"

# 所有12个目标列
ALL_TARGETS = ["zpve", "Cv", "gap", "G", "HOMO", "U", "alpha", "U0", "H", "LUMO", "mu", "R2"]

# 训练参数
TRAIN_CFG = dict(
    task='multilabel_regression',
    data_type='molecule',
    epochs=50,
    batch_size=128,
    metrics='mae',
    model_name='unimolv2',
    model_size='84m',
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=15,
    remove_hs=False,
    target_cols=[f'TARGET_{col}' for col in ALL_TARGETS],
    save_path=str(OUT_DIR / "unimol_model_all"),
    use_cuda=True,
    use_gpu='0',
)


def get_sdf_dirs(data_dir: Path) -> list:
    """获取所有包含SDF文件的目录"""
    sdf_dirs = []
    for item in data_dir.iterdir():
        if item.is_dir():
            sdf_files = list(item.glob("*.sdf"))
            if len(sdf_files) > 0:
                sdf_dirs.append(item)
    return sdf_dirs


def extract_smiles_from_sdf(mol_name, sdf_dirs):
    """从SDF文件提取SMILES"""
    for sdf_dir in sdf_dirs:
        sdf_path = sdf_dir / f"{mol_name}.sdf"
        if sdf_path.exists():
            mol = Chem.MolFromMolFile(str(sdf_path))
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                return smiles
    return None


def prepare_data():
    """准备所有12维目标的数据（原始值，归一化由unimol_tools内部处理）"""
    print("=" * 60)
    print("  准备 unimol_tools 格式数据 (12维目标)")
    print("=" * 60)
    print(f"  目标列: {ALL_TARGETS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sdf_dirs = get_sdf_dirs(DATA_DIR)

    for split in ['train', 'valid', 'test']:
        input_csv = PROCESSED_DIR / f"{split}.csv"
        output_csv = OUT_DIR / f"{split}_unimol.csv"

        if output_csv.exists():
            print(f"\n  [{split}] 使用已存在的 {output_csv}")
            df = pd.read_csv(output_csv)
            print(f"    分子数: {len(df)}")
            continue

        df = pd.read_csv(input_csv)
        print(f"\n  [{split}] 处理 {len(df)} 个分子")

        smiles_list = []
        valid_count = 0
        invalid_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  提取SMILES ({split})"):
            mol_name = str(int(row['gdb_idx']))
            smiles = extract_smiles_from_sdf(mol_name, sdf_dirs)
            if smiles is None:
                smiles = ""
                invalid_count += 1
            else:
                valid_count += 1
            smiles_list.append(smiles)

        target_dict = {f'TARGET_{col}': df[col].values for col in ALL_TARGETS}
        unimol_df = pd.DataFrame({
            'SMILES': smiles_list,
            **target_dict,
        })

        valid_mask = unimol_df['SMILES'] != ""
        unimol_df = unimol_df[valid_mask].reset_index(drop=True)

        unimol_df.to_csv(output_csv, index=False)

        print(f"\n  [{split}] 完成!")
        print(f"    有效分子: {valid_count}")
        print(f"    无效分子: {invalid_count}")
        print(f"    保存到: {output_csv}")


def train_unimol():
    """使用unimol_tools训练12维多任务模型"""
    print("\n" + "=" * 60)
    print("  使用 unimol_tools 训练 12维多任务模型")
    print("=" * 60)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_csv = str(OUT_DIR / "train_unimol.csv")
    train_df = pd.read_csv(train_csv)
    
    print(f"\n  训练数据: {len(train_df)} 个分子")
    print(f"  目标列 ({len(ALL_TARGETS)}维):")
    for i, col in enumerate(ALL_TARGETS):
        print(f"    {i+1:2d}. {col}")
    print(f"\n  模型: {TRAIN_CFG['model_name']} ({TRAIN_CFG['model_size']})")
    print(f"  批次大小: {TRAIN_CFG['batch_size']}")
    print(f"  训练轮数: {TRAIN_CFG['epochs']}")
    print(f"  学习率: {TRAIN_CFG['learning_rate']}")
    print(f"  保存路径: {TRAIN_CFG['save_path']}")
    
    # 创建训练器
    from unimol_tools import MolTrain
    
    clf = MolTrain(
        task=TRAIN_CFG['task'],
        data_type=TRAIN_CFG['data_type'],
        epochs=TRAIN_CFG['epochs'],
        batch_size=TRAIN_CFG['batch_size'],
        metrics=TRAIN_CFG['metrics'],
        model_name=TRAIN_CFG['model_name'],
        model_size=TRAIN_CFG['model_size'],
        learning_rate=TRAIN_CFG['learning_rate'],
        weight_decay=TRAIN_CFG['weight_decay'],
        patience=TRAIN_CFG['patience'],
        remove_hs=TRAIN_CFG['remove_hs'],
        target_cols=TRAIN_CFG['target_cols'],
        save_path=TRAIN_CFG['save_path'],
        use_cuda=TRAIN_CFG['use_cuda'],
        use_gpu=TRAIN_CFG['use_gpu'],
    )
    
    # 训练模型
    print("\n  开始训练...")
    pred = clf.fit(data=train_csv)
    
    print("\n  训练完成!")
    print(f"  模型保存到: {TRAIN_CFG['save_path']}")
    
    return clf


def evaluate_model():
    """评估训练好的模型（unimol_tools内部处理反归一化）"""
    print("\n" + "=" * 60)
    print("  评估 12维多任务模型")
    print("=" * 60)

    from unimol_tools import MolPredict

    clf = MolPredict(load_model=str(TRAIN_CFG['save_path']))

    for split in ['valid', 'test']:
        csv_path = str(OUT_DIR / f"{split}_unimol.csv")
        df = pd.read_csv(csv_path)

        print(f"\n  评估 {split} 集 ({len(df)} 个分子)...")

        predictions = clf.predict(data=csv_path)

        target_cols = [f'TARGET_{col}' for col in ALL_TARGETS]

        if isinstance(predictions, dict):
            pred_values = np.array([predictions[col] for col in target_cols]).T
        else:
            pred_values = predictions

        true_values = df[target_cols].values

        print(f"\n  {'目标':>8s} | {'MAE':>10s} | {'RMSE':>10s} | {'R²':>10s}")
        print("  " + "-" * 45)

        for i, col in enumerate(ALL_TARGETS):
            true = true_values[:, i]
            pred = pred_values[:, i]

            mae = np.mean(np.abs(true - pred))
            rmse = np.sqrt(np.mean((true - pred) ** 2))
            ss_res = np.sum((true - pred) ** 2)
            ss_tot = np.sum((true - true.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)

            print(f"  {col:>8s} | {mae:>10.4f} | {rmse:>10.4f} | {r2:>10.4f}")


def main():
    """主函数: 一键执行数据准备、训练、评估"""
    print("=" * 60)
    print("  Uni-Mol 12维分子性质预测")
    print("  使用 unimol_tools")
    print("=" * 60)
    
    # 步骤1: 准备数据
    prepare_data()

    # 步骤2: 训练模型
    clf = train_unimol()

    # 步骤3: 评估模型
    evaluate_model()
    
    print("\n" + "=" * 60)
    print("  全部完成!")
    print("=" * 60)
    print(f"  模型位置: {TRAIN_CFG['save_path']}")
    print(f"  数据位置: {OUT_DIR}")


if __name__ == "__main__":
    main()
