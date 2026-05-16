"""
Alchemy MVP Client - Champion
Supports: single/multi SDF prediction, test set comparison, CSV export
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
import gradio as gr
import pickle
import sys
import os
from datetime import datetime

# Add parent directory to path to access src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.champion import MoleculeGNNChampion
from src.models.enhanced import MoleculeGNNEnhanced
from src.models.gnn import MoleculeGNN
from src.dataset import (
    TARGET_COLS, BOND_TYPE_MAP, HYBRIDIZATION_MAP,
    ELECTRONEGATIVITY, ATOMIC_MASS, get_atom_feature_enhanced, get_bond_feature_enhanced
)

MODEL_TYPE = "champion"
NODE_IN_DIM = 18
EDGE_IN_DIM = 7
HIDDEN_DIM = 256
NUM_LAYERS = 5
OUT_DIM = 12
DROPOUT = 0.1

model = None
scaler = None
device = None
current_model_type = None
sdf_base_dir = Path(r"d:\Others\AIChemical\Task_one\data")
checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
test_csv_path_default = r"d:\Others\AIChemical\Task_one\data\processed\test_small.csv"

MODEL_ARCHITECTURES = {
    "champion": "MoleculeGNNChampion",
    "enhanced": "MoleculeGNNEnhanced",
    "basic": "MoleculeGNN",
}

def load_test_csv():
    """加载测试集CSV用于查找真实值"""
    global test_csv_path_default
    csv_path = Path(test_csv_path_default)
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return None
    return None

def find_actual_values_by_filename(filename):
    """根据文件名（gdb_idx）在测试集中查找真实值"""
    test_df = load_test_csv()
    if test_df is None:
        return None
    
    # 从文件名提取gdb_idx，例如 "442.sdf" -> 442
    gdb_idx_str = Path(filename).stem
    try:
        gdb_idx = int(gdb_idx_str)
    except ValueError:
        return None
    
    # 在测试集中查找
    row = test_df[test_df["gdb_idx"] == gdb_idx]
    if len(row) > 0:
        return {col: float(row[col].values[0]) for col in TARGET_COLS}
    return None

def get_checkpoint_path(model_type):
    """根据模型类型获取检查点路径"""
    if model_type == "champion":
        best_path = checkpoint_dir / "best_model_champion.pt"
        if best_path.exists():
            return str(best_path)
        legacy_path = checkpoint_dir / "best_model.pt"
        if legacy_path.exists():
            return str(legacy_path)
    elif model_type == "enhanced":
        best_path = checkpoint_dir / "best_model_enhanced.pt"
        if best_path.exists():
            return str(best_path)
        legacy_path = checkpoint_dir / "best_model.pt"
        if legacy_path.exists():
            return str(legacy_path)
    elif model_type == "basic":
        best_path = checkpoint_dir / "best_model_basic.pt"
        if best_path.exists():
            return str(best_path)
        legacy_path = checkpoint_dir / "best_model.pt"
        if legacy_path.exists():
            return str(legacy_path)
    return None

def create_model_by_type(model_type, device):
    """根据模型类型创建对应的模型架构"""
    if model_type == "champion":
        return MoleculeGNNChampion(
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            out_dim=OUT_DIM,
            dropout=DROPOUT,
        ).to(device)
    elif model_type == "enhanced":
        return MoleculeGNNEnhanced(
            hidden_dim=512,
            num_layers=8,
            out_dim=OUT_DIM,
            dropout=DROPOUT,
        ).to(device)
    elif model_type == "basic":
        return MoleculeGNN(
            node_in_dim=NODE_IN_DIM,
            edge_in_dim=EDGE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            out_dim=OUT_DIM,
            dropout=DROPOUT,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_model_and_scaler(model_path, scaler_path, model_type="auto"):
    global model, scaler, device, current_model_type

    if not Path(model_path).exists():
        return f"❌ 模型文件不存在: {model_path}"

    if not Path(scaler_path).exists():
        return f"❌ Scaler文件不存在: {scaler_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Detect model type from state_dict keys
    state_keys = checkpoint["model_state_dict"].keys()
    has_norm1 = any(k.startswith("mp_layers.0.norm1") for k in state_keys)
    has_task_heads = any(k.startswith("head.task_heads") for k in state_keys)
    has_gnn_blocks = any(k.startswith("gnn_blocks") for k in state_keys)

    if model_type == "auto":
        if has_norm1:
            model_type = "enhanced"
        elif has_task_heads:
            model_type = "champion"
        elif has_gnn_blocks:
            model_type = "basic"
        else:
            return "❌ 无法自动检测模型类型，请手动选择 enhanced, champion 或 basic"
    else:
        # User-specified type: verify against actual checkpoint
        if model_type == "enhanced" and not has_norm1:
            return "❌ 选择了 enhanced 但 checkpoint 不包含 enhanced 模型权重 (缺少 Transformer 层)，请检查模型类型"
        if model_type == "champion" and not has_task_heads:
            return "❌ 选择了 champion 但 checkpoint 不包含 champion 模型权重 (缺少 task_heads)，请检查模型类型"
        if model_type == "basic" and not has_gnn_blocks:
            return "❌ 选择了 basic 但 checkpoint 不包含 basic 模型权重 (缺少 gnn_blocks)，请检查模型类型"

    print(f"[INFO] Model type: {model_type}")
    
    model = create_model_by_type(model_type, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    current_model_type = model_type

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    param_count = sum(p.numel() for p in model.parameters())
    saved_mae = checkpoint.get("best_valid_mae", None)
    saved_epoch = checkpoint.get("epoch", None)
    mae_str = f"\n最佳MAE: {saved_mae:.6f}" if saved_mae else ""
    epoch_str = f"\n训练轮数: Epoch {saved_epoch}" if saved_epoch else ""
    return f"✅ 模型加载成功!\n设备: {device}\n模型类型: {model_type}\n参数量: {param_count:,}{mae_str}{epoch_str}"


def mol_to_graph(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_feature_enhanced(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index_list = []
    edge_attr_list = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_attr = get_bond_feature_enhanced(bond, mol)

        edge_index_list.append([i, j])
        edge_attr_list.append(edge_attr)
        edge_index_list.append([j, i])
        edge_attr_list.append(edge_attr)

    if len(edge_index_list) > 0:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_IN_DIM), dtype=torch.float)

    conf = mol.GetConformer()
    if conf is not None:
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    else:
        pos = torch.zeros((x.size(0), 3), dtype=torch.float)

    return x, edge_index, edge_attr, pos


def predict_single_molecule(mol, filename=""):
    global model, scaler, device

    if model is None or scaler is None:
        return None, "❌ 请先加载模型和Scaler!"

    try:
        x, edge_index, edge_attr, pos = mol_to_graph(mol)

        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos,
                    batch=torch.zeros(x.size(0), dtype=torch.long))
        data = data.to(device)

        with torch.no_grad():
            pred_scaled = model(data).cpu().numpy().flatten()

        pred_real = scaler.inverse_transform(pred_scaled.reshape(1, -1)).flatten()

        result_dict = {col: float(val) for col, val in zip(TARGET_COLS, pred_real)}
        return result_dict, f"✅ 预测成功: {filename}"

    except Exception as e:
        import traceback
        return None, f"❌ 预测失败 {filename}: {str(e)}\n{traceback.format_exc()}"


def predict_from_sdf_content(sdf_content, sdf_filename):
    try:
        mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
        if mol is None:
            return None, f"❌ 无法解析SDF文件: {sdf_filename}"

        return predict_single_molecule(mol, sdf_filename)

    except Exception as e:
        return None, f"❌ 预测失败 {sdf_filename}: {str(e)}"


def predict_single_sdf(sdf_file):
    if sdf_file is None:
        return None, "❌ 请上传SDF文件!"

    with open(sdf_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    pred_dict, msg = predict_from_sdf_content(content, Path(sdf_file).name)
    
    if pred_dict is None:
        return None, msg
    
    # 转换为DataFrame以便Gradio正确显示
    result_row = {"filename": Path(sdf_file).name}
    
    # 尝试查找真实值
    actual_values = find_actual_values_by_filename(Path(sdf_file).name)
    
    if actual_values is not None:
        # 有真实值，做对比
        for col in TARGET_COLS:
            result_row[f"{col}_actual"] = actual_values[col]
            result_row[f"{col}_pred"] = pred_dict[col]
            result_row[f"{col}_error"] = abs(pred_dict[col] - actual_values[col])
        df = pd.DataFrame([result_row])
        # 按 实际值/预测值/误差 的顺序排列
        cols = ["filename"]
        for col in TARGET_COLS:
            cols.extend([f"{col}_actual", f"{col}_pred", f"{col}_error"])
        df = df[cols]
        status = f"✅ 预测成功: {Path(sdf_file).name}\n📊 已加载真实值，显示预测 vs 实际对比"
    else:
        # 没有真实值，只显示预测值
        for col in TARGET_COLS:
            result_row[col] = pred_dict[col]
        df = pd.DataFrame([result_row])
        df = df[["filename"] + TARGET_COLS]
        status = f"✅ 预测成功: {Path(sdf_file).name}\n⚠️ 未在测试集中找到对应分子，仅显示预测值"
    
    return df, status


def predict_multiple_sdf(sdf_files):
    if sdf_files is None or len(sdf_files) == 0:
        return None, "❌ 请上传SDF文件!"

    results = []
    errors = []

    for sdf_file in sdf_files:
        try:
            with open(sdf_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            pred_dict, msg = predict_from_sdf_content(content, Path(sdf_file).name)
            if pred_dict is not None:
                result_row = {"filename": Path(sdf_file).name}
                
                # 尝试查找真实值
                actual_values = find_actual_values_by_filename(Path(sdf_file).name)
                
                if actual_values is not None:
                    # 有真实值，做对比
                    for col in TARGET_COLS:
                        result_row[f"{col}_actual"] = actual_values[col]
                        result_row[f"{col}_pred"] = pred_dict[col]
                        result_row[f"{col}_error"] = abs(pred_dict[col] - actual_values[col])
                else:
                    # 没有真实值，只显示预测值
                    for col in TARGET_COLS:
                        result_row[col] = pred_dict[col]
                
                results.append(result_row)
            else:
                errors.append(msg)
        except Exception as e:
            errors.append(f"❌ 处理失败 {Path(sdf_file).name}: {str(e)}")

    if len(results) == 0:
        return None, "\n".join(errors) if errors else "❌ 所有文件预测失败!"

    df = pd.DataFrame(results)
    
    # 根据是否有真实值，调整列顺序
    if actual_values is not None:
        # 有真实值，按 实际值/预测值/误差 的顺序排列
        cols = ["filename"]
        for col in TARGET_COLS:
            cols.extend([f"{col}_actual", f"{col}_pred", f"{col}_error"])
        df = df[cols]
    else:
        # 没有真实值，只显示预测值
        df = df[["filename"] + TARGET_COLS]

    status = f"✅ 成功预测 {len(results)} 个分子"
    if errors:
        status += f"\n❌ 失败 {len(errors)} 个:\n" + "\n".join(errors)
    
    if actual_values is not None:
        status += "\n📊 已加载真实值，显示预测 vs 实际对比"
    else:
        status += "\n⚠️ 未在测试集中找到对应分子，仅显示预测值"

    return df, status


def find_sdf_by_idx(gdb_idx):
    for sdf_dir in sdf_base_dir.iterdir():
        if sdf_dir.is_dir():
            sdf_path = sdf_dir / f"{gdb_idx}.sdf"
            if sdf_path.exists():
                return sdf_path
    return None


def predict_test_set(test_csv_path):
    global model, scaler, device

    if model is None or scaler is None:
        return None, None, "❌ 请先加载模型和Scaler!"

    if not Path(test_csv_path).exists():
        return None, None, f"❌ 测试集文件不存在: {test_csv_path}"

    test_df = pd.read_csv(test_csv_path)
    
    results = []
    errors = []
    success_count = 0

    for idx, row in test_df.iterrows():
        gdb_idx = int(row["gdb_idx"])
        actual_values = {col: row[col] for col in TARGET_COLS}
        
        sdf_path = find_sdf_by_idx(gdb_idx)
        
        if sdf_path is None:
            errors.append(f"❌ 未找到SDF文件: {gdb_idx}.sdf")
            continue
        
        try:
            with open(sdf_path, "r", encoding="utf-8", errors="ignore") as f:
                sdf_content = f.read()
            
            mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)
            if mol is None:
                mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
            
            if mol is None:
                errors.append(f"❌ 无法解析SDF: {gdb_idx}.sdf")
                continue

            pred_dict, msg = predict_single_molecule(mol, f"{gdb_idx}.sdf")
            
            if pred_dict is not None:
                result_row = {
                    "gdb_idx": gdb_idx,
                    "sdf_file": f"{gdb_idx}.sdf",
                }
                
                for col in TARGET_COLS:
                    result_row[f"{col}_actual"] = actual_values[col]
                    result_row[f"{col}_pred"] = pred_dict[col]
                    result_row[f"{col}_error"] = abs(pred_dict[col] - actual_values[col])
                
                results.append(result_row)
                success_count += 1
            else:
                errors.append(f"❌ 预测失败 {gdb_idx}: {msg}")
                
        except Exception as e:
            errors.append(f"❌ 处理失败 {gdb_idx}: {str(e)}")

    if len(results) == 0:
        return None, None, "\n".join(errors) if errors else "❌ 所有分子预测失败!"

    result_df = pd.DataFrame(results)
    
    mae_per_target = {}
    for col in TARGET_COLS:
        mae_per_target[col] = result_df[f"{col}_error"].mean()
    
    overall_mae = np.mean(list(mae_per_target.values()))
    
    summary_stats = []
    for col in TARGET_COLS:
        summary_stats.append({
            "Property": col,
            "MAE": mae_per_target[col],
            "Min Error": result_df[f"{col}_error"].min(),
            "Max Error": result_df[f"{col}_error"].max(),
            "Mean Actual": result_df[f"{col}_actual"].mean(),
            "Mean Predicted": result_df[f"{col}_pred"].mean(),
        })
    
    stats_df = pd.DataFrame(summary_stats)
    
    status = f"✅ 成功预测 {success_count}/{len(test_df)} 个分子\n"
    status += f"📊 整体MAE: {overall_mae:.6f}\n\n"
    status += "各属性MAE:\n"
    for col in TARGET_COLS:
        status += f"  {col}: {mae_per_target[col]:.6f}\n"
    
    if errors:
        status += f"\n❌ 失败 {len(errors)} 个:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            status += f"\n... 还有 {len(errors) - 10} 个错误"

    return result_df, stats_df, status


def export_results(df, stats_df=None):
    if df is None:
        return None, None, "❌ 没有可导出的数据!"

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pred_path = output_dir / f"predictions_{timestamp}.csv"
    df.to_csv(pred_path, index=False)
    
    stats_path = None
    if stats_df is not None:
        stats_path = output_dir / f"statistics_{timestamp}.csv"
        stats_df.to_csv(stats_path, index=False)
    
    return str(pred_path), str(stats_path) if stats_path else None, f"✅ 已导出预测结果到: {pred_path}"


def update_model_path(model_type):
    """根据模型类型自动更新模型路径"""
    if model_type == "auto":
        auto_path = get_checkpoint_path("champion")
        return auto_path if auto_path else ""
    elif model_type == "champion":
        path = get_checkpoint_path("champion")
        return path if path else ""
    elif model_type == "enhanced":
        path = get_checkpoint_path("enhanced")
        return path if path else ""
    elif model_type == "basic":
        path = get_checkpoint_path("basic")
        return path if path else ""
    return ""

with gr.Blocks(title="Alchemy 分子性质预测系统", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧪 Alchemy 分子性质预测系统")
    gr.Markdown("使用冠军版图神经网络预测12个量子化学性质 | ape-MPNN算法")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. 加载模型")
            model_type = gr.Dropdown(
                choices=["auto", "champion", "enhanced", "basic"],
                value="auto",
                label="模型类型",
                info="auto: 自动检测 | champion: 冠军版 | enhanced: Transformer增强版 | basic: 基础版"
            )
            model_path = gr.Textbox(
                label="模型路径",
                value="",
                placeholder="选择模型类型后自动填充..."
            )
            scaler_path = gr.Textbox(
                label="Scaler路径",
                value=str(Path(__file__).parent.parent / "outputs" / "target_scaler.pkl"),
                placeholder="选择Scaler文件路径..."
            )
            load_btn = gr.Button("📥 加载模型", variant="primary")
            load_status = gr.Textbox(label="加载状态", interactive=False, lines=3)

        with gr.Column():
            gr.Markdown("### 2. 预测结果")
            result_df = gr.DataFrame(
                label="预测结果",
                interactive=False
            )
            predict_status = gr.Textbox(label="预测状态", interactive=False, lines=3)

    gr.Markdown("---")
    gr.Markdown("### 3. SDF文件预测")

    with gr.Tab("📄 单文件预测"):
        single_sdf = gr.File(label="上传单个SDF文件", file_types=[".sdf"])
        single_pred_btn = gr.Button("🔮 预测", variant="primary")

    with gr.Tab("📁 批量预测"):
        multi_sdf = gr.File(label="上传多个SDF文件", file_count="multiple", file_types=[".sdf"])
        multi_pred_btn = gr.Button("🔮 批量预测", variant="primary")

    with gr.Tab("📊 测试集对比"):
        gr.Markdown("### 测试集预测与对比")
        gr.Markdown("自动从data目录查找对应SDF文件，预测并与真实值对比")
        
        test_csv_path = gr.Textbox(
            label="测试集CSV路径",
            value=r"d:\Others\AIChemical\Task_one\data\processed\test_small.csv",
            placeholder="选择测试集CSV文件路径..."
        )
        test_pred_btn = gr.Button("🚀 开始测试集预测", variant="primary")
        
        with gr.Row():
            with gr.Column():
                test_result_df = gr.DataFrame(
                    label="预测 vs 实际值对比",
                    interactive=False
                )
            with gr.Column():
                test_stats_df = gr.DataFrame(
                    label="统计指标",
                    interactive=False
                )
        
        test_status = gr.Textbox(label="测试状态", interactive=False, lines=8)

    gr.Markdown("---")
    gr.Markdown("### 4. 导出结果")
    
    with gr.Row():
        export_btn = gr.Button("💾 导出为CSV", variant="primary")
        export_test_btn = gr.Button("💾 导出测试集结果", variant="primary")
    
    with gr.Row():
        export_status = gr.Textbox(label="导出状态", interactive=False)
        export_path = gr.Textbox(label="预测文件路径", interactive=False)
        stats_path = gr.Textbox(label="统计文件路径", interactive=False)

    model_type.change(
        fn=update_model_path,
        inputs=[model_type],
        outputs=[model_path]
    )

    load_btn.click(
        fn=load_model_and_scaler,
        inputs=[model_path, scaler_path, model_type],
        outputs=[load_status]
    )

    single_pred_btn.click(
        fn=predict_single_sdf,
        inputs=[single_sdf],
        outputs=[result_df, predict_status]
    )

    multi_pred_btn.click(
        fn=predict_multiple_sdf,
        inputs=[multi_sdf],
        outputs=[result_df, predict_status]
    )

    test_pred_btn.click(
        fn=predict_test_set,
        inputs=[test_csv_path],
        outputs=[test_result_df, test_stats_df, test_status]
    )

    export_btn.click(
        fn=export_results,
        inputs=[result_df],
        outputs=[export_path, stats_path, export_status]
    )

    export_test_btn.click(
        fn=export_results,
        inputs=[test_result_df, test_stats_df],
        outputs=[export_path, stats_path, export_status]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
