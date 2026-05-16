"""
使用 unimol_tools 对 mu 和 R2 数据进行回归训练

unimol_tools 是一个基于 Uni-Mol 的分子性质预测工具，支持：
- 自动从 SMILES 提取分子表示
- 多任务回归/分类
- 内置预训练模型

运行方式:
  python -m alchemy_mvp.src.unimol.train_with_unimol
  或
  python alchemy_mvp/src/unimol/train_with_unimol.py
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
from unimol_tools import MolTrain, MolPredict

# ── 配置 ──
DATA_DIR = _PROJECT.parent / "data"
OUT_DIR = _PROJECT / "output_unimol"
PROCESSED_DIR = DATA_DIR / "processed"

# 目标列
TARGET_COLS = ["mu", "R2"]

# 训练参数
TRAIN_CFG = dict(
    task='regression',
    data_type='molecule',
    epochs=50,
    batch_size=32,
    metrics='mae',
    model_name='unimolv2',      # 使用 Uni-Mol V2
    model_size='84m',           # 84M 参数版本
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=15,
    remove_hs=False,
    target_cols=[f'TARGET_{col}' for col in TARGET_COLS],
    save_path=str(OUT_DIR / "unimol_model"),
)


def prepare_data():
    """准备 unimol_tools 格式的数据。
    
    unimol_tools 需要 CSV 格式：
    - SMILES 列：分子 SMILES 字符串
    - TARGET_xxx 列：目标值（回归）或 0/1（分类）
    """
    print("=" * 60)
    print("  准备 unimol_tools 格式数据")
    print("=" * 60)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        input_csv = PROCESSED_DIR / f"{split}.csv"
        output_csv = OUT_DIR / f"{split}_unimol.csv"
        
        if output_csv.exists():
            print(f"  {split}: 使用已存在的 {output_csv}")
            continue
        
        df = pd.read_csv(input_csv)
        print(f"\n  处理 {split} 分割: {len(df)} 个分子")
        
        # 从 SDF 文件提取 SMILES
        from rdkit import Chem
        from src.utils import get_sdf_dirs
        
        sdf_dirs = get_sdf_dirs(DATA_DIR)
        smiles_list = []
        
        for idx, row in df.iterrows():
            mol_name = str(int(row['gdb_idx']))
            smiles = None
            
            for sdf_dir in sdf_dirs:
                sdf_path = sdf_dir / f"{mol_name}.sdf"
                if sdf_path.exists():
                    mol = Chem.MolFromMolFile(str(sdf_path))
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                        break
            
            if smiles is None:
                smiles = ""  # 标记为无效
            smiles_list.append(smiles)
        
        # 创建 unimol_tools 格式 DataFrame
        unimol_df = pd.DataFrame({
            'SMILES': smiles_list,
            'TARGET_mu': df['mu'].values,
            'TARGET_R2': df['R2'].values,
        })
        
        # 过滤掉无效 SMILES
        valid_mask = unimol_df['SMILES'] != ""
        unimol_df = unimol_df[valid_mask].reset_index(drop=True)
        
        unimol_df.to_csv(output_csv, index=False)
        print(f"  保存 {len(unimol_df)} 个有效分子到 {output_csv}")
        print(f"  过滤掉 {(~valid_mask).sum()} 个无效分子")


def train_unimol():
    """使用 unimol_tools 训练模型。"""
    print("\n" + "=" * 60)
    print("  使用 unimol_tools 训练 Uni-Mol 模型")
    print("=" * 60)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载训练数据
    train_csv = str(OUT_DIR / "train_unimol.csv")
    valid_csv = str(OUT_DIR / "valid_unimol.csv")
    
    train_df = pd.read_csv(train_csv)
    print(f"\n  训练数据: {len(train_df)} 个分子")
    print(f"  目标列: {TRAIN_CFG['target_cols']}")
    print(f"  模型: {TRAIN_CFG['model_name']} ({TRAIN_CFG['model_size']})")
    print(f"  批次大小: {TRAIN_CFG['batch_size']}")
    print(f"  训练轮数: {TRAIN_CFG['epochs']}")
    print(f"  学习率: {TRAIN_CFG['learning_rate']}")
    
    # 创建训练器
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
    )
    
    # 训练模型
    print("\n  开始训练...")
    pred = clf.fit(data=train_csv)
    
    print("\n  训练完成！")
    print(f"  模型保存到: {TRAIN_CFG['save_path']}")
    
    return clf


def evaluate_model():
    """评估训练好的模型。"""
    print("\n" + "=" * 60)
    print("  评估模型")
    print("=" * 60)
    
    # 加载预测器
    clf = MolPredict(load_model=str(TRAIN_CFG['save_path']))
    
    for split in ['valid', 'test']:
        csv_path = str(OUT_DIR / f"{split}_unimol.csv")
        df = pd.read_csv(csv_path)
        
        print(f"\n  评估 {split} 集 ({len(df)} 个分子)...")
        
        # 预测
        predictions = clf.predict(data=csv_path)
        
        # 计算指标
        target_cols = [f'TARGET_{col}' for col in TARGET_COLS]
        true_values = df[target_cols].values
        
        # 处理 predictions 格式
        if isinstance(predictions, dict):
            pred_values = np.array([predictions[col] for col in target_cols]).T
        else:
            pred_values = predictions
        
        # 计算 MAE, RMSE, R²
        for i, col in enumerate(TARGET_COLS):
            true = true_values[:, i]
            pred = pred_values[:, i]
            
            mae = np.mean(np.abs(true - pred))
            rmse = np.sqrt(np.mean((true - pred) ** 2))
            r2 = 1 - np.sum((true - pred) ** 2) / (np.sum((true - true.mean()) ** 2) + 1e-8)
            
            print(f"    {col:>4s}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")


def main():
    """主函数：一键执行数据准备、训练、评估。"""
    print("=" * 60)
    print("  Uni-Mol 分子性质预测 (mu, R2)")
    print("  使用 unimol_tools")
    print("=" * 60)
    
    # 步骤 1: 准备数据
    prepare_data()
    
    # 步骤 2: 训练模型
    clf = train_unimol()
    
    # 步骤 3: 评估模型
    evaluate_model()
    
    print("\n" + "=" * 60)
    print("  全部完成！")
    print("=" * 60)
    print(f"  模型位置: {TRAIN_CFG['save_path']}")
    print(f"  数据位置: {OUT_DIR}")


if __name__ == "__main__":
    main()
