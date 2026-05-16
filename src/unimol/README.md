# Uni-Mol+ for Molecular Property Prediction

基于Uni-Mol+架构的分子性质预测模型，用于预测量子化学性质（mu, R2等）。

## 架构

```
SMILES -> RDKit 3D构象 -> Uni-Mol+ Transformer -> 性质预测
```

### 核心组件

1. **ConformerGenerator**: 使用RDKit从SMILES生成3D构象
2. **AtomEmbedding**: 原子序数嵌入
3. **DistanceEmbedding**: 距离编码（高斯RBF）
4. **MultiHeadAttention**: 带3D偏置的多头注意力
5. **TransformerEncoder**: 带3D偏置的Transformer编码器
6. **PropertyHead**: 性质预测头

## 安装依赖

```bash
pip install rdkit torch torch-geometric numpy pandas scikit-learn tqdm
```

## 使用方法

### 从项目根目录运行

```bash
# 方式1：作为模块运行
python -m alchemy_mvp.src.unimol.train_unimol

# 方式2：直接运行
python alchemy_mvp/src/unimol/train_unimol.py
```

### 配置参数

在 `train_unimol.py` 中的 `CFG` 字典可以修改：

```python
CFG = dict(
    hidden_dim     = 256,        # 隐藏层维度
    num_layers     = 6,          # Transformer层数
    num_heads      = 8,          # 注意力头数
    ffn_dim        = 1024,       # FFN维度
    dropout        = 0.1,        # Dropout率
    batch_size     = 32,         # 批次大小
    epochs         = 200,        # 训练轮数
    lr             = 1e-4,       # 学习率
    weight_decay   = 1e-4,       # 权重衰减
    patience       = 40,         # 早停耐心值
)
```

## 输出

训练完成后，在 `alchemy_mvp/output_unimol/` 目录下生成：

- `model.pt` - 最佳模型权重
- `training.log` - 训练日志（CSV格式）

## 与GNN模型的对比

| 特性 | GNN (gnn_mu_r2.py) | Uni-Mol+ (train_unimol.py) |
|------|-------------------|---------------------------|
| 架构 | PaiNN等变GNN | Transformer + 3D偏置 |
| 输入 | 分子图 + 坐标 | SMILES -> 3D构象 |
| 消息传递 | 等变向量消息 | 注意力 + 距离偏置 |
| 适用场景 | 需要预计算图 | 端到端从SMILES |

## 注意事项

1. **3D构象生成**: 使用RDKit的ETKDG方法，可能需要较长时间
2. **内存占用**: Transformer的注意力机制是O(N²)，大分子需要注意
3. **数据缓存**: 构象会在内存中缓存，避免重复计算
