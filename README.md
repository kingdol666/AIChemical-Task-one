# Alchemy MVP - Molecular Property Prediction

基于图神经网络（GNN）的分子性质预测MVP项目，使用SDF分子结构预测12个量子化学性质。

## 项目结构

```
alchemy_mvp/
├── requirements.txt              # Python依赖
├── README.md                     # 项目说明
├── run.py                        # 一键启动入口
├── src/
│   ├── split_data.py             # 数据分割和标准化
│   ├── dataset.py                # PyTorch Geometric数据集
│   ├── model.py                  # 标准GNN模型
│   ├── model_best.py             # 冠军算法模型 (ape-MPNN)
│   ├── train.py                  # 单模型训练脚本
│   ├── train_ensemble.py         # 多模型集成训练脚本
│   ├── predict.py                # 预测脚本
│   └── utils.py                  # 工具函数
├── client/                       # 客户端应用
│   ├── app.py                    # Gradio Web界面
│   └── requirements.txt          # 客户端依赖
├── outputs/                      # 输出目录
│   ├── target_scaler.pkl         # 拟合的StandardScaler
│   └── answer.csv                # 最终预测结果
├── checkpoints/                  # 模型检查点
│   ├── best_model_champion.pt    # 冠军模型最佳权重
│   ├── last_checkpoint_champion.pt # 冠军模型最新检查点
│   └── model_1_champion/         # 集成训练的冠军模型检查点
└── cache/                        # 数据缓存
    └── dataset_*.pt              # 预处理的分子图数据
```

## 安装

### 环境要求

- Python 3.8+
- CUDA 11.8+ (可选，用于GPU加速)

### 安装依赖

```bash
cd alchemy_mvp
pip install -r requirements.txt
```

**PyTorch安装说明：**

CPU版本：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

CUDA版本（推荐）：

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

## 数据准备

将数据放在 `data/` 目录下：

1. **CSV文件**：包含分子性质的CSV文件，列包括：
   - `gdb_idx`：分子ID（必需）
   - `atom number`：原子数（可选）
   - 12个目标性质：`zpve`, `Cv`, `gap`, `G`, `HOMO`, `U`, `alpha`, `U0`, `H`, `LUMO`, `mu`, `R2`

2. **SDF文件**：分子结构文件，按子目录组织（如 `data/atom_10/`, `data/atom_11/` 等）。每个SDF文件应以对应的 `gdb_idx` 命名（如 `123.sdf`）。

## 使用方法

### 方式一：一键运行（推荐）

运行完整流程（数据分割 → 训练 → 预测）：

```bash
# 标准训练
python run.py --epochs 120 --batch_size 128 --lr 5e-4

# 断点续训
python run.py --epochs 120 --batch_size 128 --lr 5e-4 --resume

# 集成训练（多模型）
python run.py --ensemble --epochs 120 --batch_size 128

# 集成训练续训
python run.py --ensemble --epochs 120 --batch_size 128 --resume
```

### 方式二：自定义参数

#### 1. 标准训练（单模型）

```bash
# 使用默认参数（champion模型）
python run.py

# 指定模型类型
python run.py --model_type champion

# 完整参数示例
python run.py --epochs 100 --batch_size 64 --lr 5e-4 --device cuda --model_type champion

# 断点续训
python run.py --resume --model_type champion
```

**参数说明：**

| 参数            | 类型  | 默认值     | 说明                         |
| --------------- | ----- | ---------- | ---------------------------- |
| `--epochs`      | int   | 50         | 训练轮数                     |
| `--batch_size`  | int   | 64         | 批次大小                     |
| `--lr`          | float | 1e-3       | 学习率                       |
| `--device`      | str   | auto       | 设备：auto, cpu, cuda        |
| `--model_type`  | str   | champion   | 模型架构：champion, standard |
| `--ensemble`    | flag  | False      | 启用集成训练模式             |
| `--model_types` | list  | [champion] | 集成训练的模型类型列表       |
| `--acsf_weight` | float | 0.01       | ACSF正则化权重               |
| `--resume`      | flag  | False      | 从上次检查点继续训练         |

**模型类型说明：**

| 模型类型   | 架构            | 特点                       | 适用场景           |
| ---------- | --------------- | -------------------------- | ------------------ |
| `standard` | GNN+Transformer | 基础模型，训练速度快       | 快速验证、基线对比 |
| `champion` | ape-MPNN        | 竞赛冠军算法，LSTM消息传递 | 高精度预测（推荐） |

#### 2. 集成训练（多模型）

```bash
# 通过 run.py 启用集成训练（推荐）
python run.py --ensemble

# 指定模型组合
python run.py --ensemble --model_types champion champion

# 完整参数示例
python run.py --ensemble --epochs 120 --batch_size 128 --lr 5e-4 --device cuda --model_types champion champion --acsf_weight 0.01

# 断点续训
python run.py --ensemble --resume --model_types champion
```

**或者直接调用集成训练脚本：**

```bash
# 使用默认模型组合（champion）
python src/train_ensemble.py

# 指定模型组合
python src/train_ensemble.py --model_types champion champion

# 完整参数示例
python src/train_ensemble.py --epochs 120 --batch_size 128 --lr 5e-4 --device cuda --model_types champion champion --acsf_weight 0.01

# 断点续训
python src/train_ensemble.py --resume --model_types champion
```

**集成训练参数说明：**

| 参数            | 类型  | 默认值     | 说明                  |
| --------------- | ----- | ---------- | --------------------- |
| `--epochs`      | int   | 120        | 每个模型的训练轮数    |
| `--batch_size`  | int   | 128        | 批次大小              |
| `--lr`          | float | 5e-4       | 学习率                |
| `--device`      | str   | auto       | 设备：auto, cpu, cuda |
| `--model_types` | list  | [champion] | 要训练的模型类型列表  |
| `--acsf_weight` | float | 0.01       | ACSF正则化权重        |
| `--resume`      | flag  | False      | 从上次检查点继续训练  |

#### 3. 仅预测

```bash
python src/predict.py --device cuda
```

### 方式三：分步执行

1. **分割数据并拟合Scaler：**

   ```bash
   python src/split_data.py
   ```

2. **训练模型：**

   ```bash
   python src/train.py --epochs 50 --batch_size 64 --lr 1e-3 --model_type champion
   ```

3. **生成预测：**
   ```bash
   python src/predict.py
   ```

### 启动客户端应用

```bash
cd client
pip install -r requirements.txt
python app.py
```

客户端启动后，浏览器会自动打开Web界面，支持：

- 上传单个/多个SDF文件进行预测
- 加载测试集进行模型评估
- 导出预测结果为CSV
- 对比预测值与真实值

## 输出文件

| 文件                             | 说明                 |
| -------------------------------- | -------------------- |
| `data/processed/train.csv`       | 训练集（80%）        |
| `data/processed/valid.csv`       | 验证集（10%）        |
| `data/processed/test.csv`        | 测试集（10%）        |
| `outputs/target_scaler.pkl`      | 拟合的StandardScaler |
| `checkpoints/best_model.pt`      | 最佳模型检查点       |
| `checkpoints/last_checkpoint.pt` | 最新训练检查点       |
| `outputs/answer.csv`             | 测试集最终预测结果   |

## answer.csv 格式

输出文件 `outputs/answer.csv` 包含12个量子化学性质的预测：

```csv
gdb_idx,zpve,Cv,gap,G,HOMO,U,alpha,U0,H,LUMO,mu,R2
123,0.123,45.6,0.456,-123.4,-0.456,123.4,56.7,-123.0,-123.2,-0.123,1.234,567.8
...
```

- 按 `gdb_idx` 升序排序
- 所有值为原始物理单位（反标准化后）
- 无NaN值

## 模型架构

### 冠军模型 (champion)

- **算法**：ape-MPNN（Alchemy竞赛冠军算法）
- **特点**：LSTM消息传递、多级注意力池化、ACSF几何编码、多任务学习头
- **隐藏维度**：256
- **层数**：5层
- **节点特征**：原子序数、度、形式电荷、杂化、氢数、芳香性、环信息、手性、重原子数、金属标志、价电子数（18维）
- **边特征**：键类型、立体化学、芳香性、共轭、环信息、RBF距离编码（7维）

### 标准模型 (standard)

- **图神经网络**：GINEConv + 多尺度池化
- **节点特征**：原子序数、度、形式电荷、芳香性、杂化、总氢数、电负性、原子质量、自由基电子、环信息、手性（18维）
- **边特征**：单键、双键、三键、芳香键、共轭、环信息、键长（7维）
- **隐藏维度**：256
- **输出**：12维回归

## 训练配置

- **框架**：PyTorch + PyTorch Geometric + RDKit
- **损失函数**：Per-Target加权SmoothL1Loss
- **优化器**：AdamW (weight_decay=1e-2)
- **学习率调度**：LinearLR Warmup + ReduceLROnPlateau（论文策略：val MAE 连续12轮不降则LR减半）
- **数据分割**：80% 训练 / 10% 验证 / 10% 测试（固定随机种子=42）
- **标准化**：StandardScaler（仅在训练集上拟合）
- **评估指标**：MAE（标准化空间和真实物理尺度）
- **早停策略**：50轮耐心值，配合ReduceLROnPlateau自适应降低学习率

## 技术细节

### 特征工程

所有特征均基于化学和物理规律，非凭空编造：

**节点特征（18维）：**

1. 原子序数（One-hot，1-10）
2. 度（One-hot，0-4）
3. 形式电荷
4. 芳香性标志
5. 杂化类型（One-hot）
6. 总氢数
7. 电负性（基于Pauling标度）
8. 原子质量（归一化）
9. 自由基电子数
10. 环成员标志
11. 手性标志

**边特征（7维）：**

1. 单键、双键、三键、芳香键（One-hot）
2. 共轭标志
3. 环键标志
4. 键长（高斯RBF编码）

### 3D几何编码

冠军模型使用：

- 高斯RBF距离编码（50个基函数）
- ACSF（原子簇对称函数）正则化（32个基函数）

## 常见问题

### Q: 训练太慢怎么办？

A: 建议使用CUDA GPU，并设置合适的batch_size（128或256）。首次运行会缓存分子图数据，后续训练会直接加载缓存。

### Q: 如何断点续训？

A: 添加 `--resume` 参数即可从上次检查点继续训练。

### Q: 如何选择模型？

A:

- 快速验证：使用 `standard` 模型
- 高精度：使用 `champion` 模型（推荐）
- 最佳效果：使用集成训练 `train_ensemble.py` 训练多个 champion 模型

### Q: 客户端如何使用？

A: 运行 `python client/app.py`，在Web界面中上传SDF文件或加载测试集，点击预测即可。

## 许可证

本项目仅供学习和研究使用。
