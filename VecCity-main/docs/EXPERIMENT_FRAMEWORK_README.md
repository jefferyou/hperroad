# HRNR_Hyperbolic 完整实验框架

## 概览

本实验框架为HRNR_Hyperbolic模型提供了完整的训练、评估、超参数优化和可视化工具，支持多种实验模式和分析方法。

## 🚀 快速开始

### 环境要求

```bash
# Python依赖
pip install torch numpy scipy scikit-learn matplotlib seaborn

# 可选：用于超参数优化
pip install hyperopt GPyOpt
```

### 最简单的运行方式

```bash
cd experiments
./quick_start.sh --mode single --dataset xa
```

## 📁 文件结构

```
hperroad/
├── VecCity-main/                          # VecCity框架
│   ├── veccity/
│   │   ├── config/
│   │   │   ├── model/segment/
│   │   │   │   ├── HRNR_Hyperbolic.json         # 模型配置
│   │   │   ├── data/
│   │   │   │   └── HRNR_HyperbolicDataset.json  # 数据配置
│   │   │   └── executor/
│   │   │       └── HRNR_HyperbolicExecutor.json # 执行器配置
│   │   ├── upstream/road_representation/
│   │   │   ├── HRNR_Hyperbolic.py              # 主模型
│   │   │   ├── hyperbolic_utils.py             # 双曲工具
│   │   │   └── __init__.py                     # 导出
│   │   └── pipeline.py                          # 实验pipeline
│   └── run_model.py                             # VecCity入口
│
├── experiments/                           # 实验脚本目录
│   ├── README.md                          # 实验指南
│   ├── run_hrnr_hyperbolic.py            # 主实验脚本
│   ├── hyperparameter_tuning.py          # 超参数优化
│   ├── visualization_tools.py             # 可视化工具
│   ├── example_usage.py                   # 使用示例
│   ├── quick_start.sh                     # 快速启动脚本
│   ├── results/                           # 结果目录
│   └── figures/                           # 图片目录
│
├── HRNR_HYPERBOLIC_README.md             # 模型技术文档
└── EXPERIMENT_FRAMEWORK_README.md        # 本文件
```

## 🎯 核心功能

### 1. 实验运行模式

| 模式 | 命令 | 说明 |
|------|------|------|
| **单次实验** | `--mode single` | 运行一次完整的训练和评估 |
| **多随机种子** | `--mode multi_seed` | 用不同种子运行多次，评估稳定性 |
| **消融实验** | `--mode ablation` | 测试各组件的贡献 |
| **模型对比** | `--mode comparison` | 对比HRNR和HRNR_Hyperbolic |
| **超参数优化** | `hyperparameter_tuning.py` | 自动搜索最佳超参数 |

### 2. 超参数优化方法

支持三种搜索策略：

- **Random Search（随机搜索）**: 推荐，快速且有效
- **Grid Search（网格搜索）**: 全面但耗时
- **Bayesian Optimization（贝叶斯优化）**: 智能搜索（需要额外库）

### 3. 可视化分析

- 训练曲线（损失、AUC、F1等）
- 超参数重要性分析
- 消融实验对比
- 模型性能对比
- 双曲嵌入PCA可视化

## 📊 使用示例

### 示例1: 基础训练

```bash
# 使用shell脚本
cd experiments
./quick_start.sh --mode single --dataset xa --gpu 0

# 或使用Python脚本
python run_hrnr_hyperbolic.py \
    --dataset xa \
    --seed 0 \
    --hyperbolic_dim 224 \
    --lambda_ce 0.1 \
    --lambda_cc 0.1
```

### 示例2: 超参数优化

```bash
# 随机搜索50次
python hyperparameter_tuning.py \
    --method random \
    --max_trials 50 \
    --dataset xa \
    --metric auc

# 使用自定义搜索空间
python hyperparameter_tuning.py \
    --search_space_file custom_search_space.json \
    --method grid \
    --max_trials 100
```

### 示例3: 消融实验

```bash
# 自动运行所有消融配置
./quick_start.sh --mode ablation --dataset xa

# 或使用Python
python run_hrnr_hyperbolic.py \
    --mode ablation \
    --dataset xa
```

结果将包括：
- 完整模型（蕴含 + 对比）
- 无蕴含损失
- 无对比损失
- 仅结构损失

### 示例4: 多随机种子评估

```bash
# 5个不同随机种子
./quick_start.sh --mode multi_seed --dataset xa

# 自定义运行次数
python run_hrnr_hyperbolic.py \
    --mode multi_seed \
    --num_runs 10 \
    --dataset xa
```

### 示例5: 可视化结果

```python
from visualization_tools import ExperimentVisualizer

visualizer = ExperimentVisualizer()

# 训练曲线
visualizer.plot_training_curves('veccity/log/exp.log')

# 超参数分析
visualizer.plot_hyperparameter_importance('results/hypertuning_*.json')

# 消融实验
visualizer.plot_ablation_study('results/*_ablation_study.json')

# 嵌入可视化
visualizer.plot_embedding_pca('veccity/cache/exp/evaluate_cache/road_embedding.npy')
```

## 🔧 配置系统

### 三层配置结构

1. **默认配置**: 在`veccity/config/`中定义
2. **配置文件**: JSON格式的自定义配置
3. **命令行参数**: 最高优先级

### 关键参数说明

#### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hyperbolic_dim` | int | 224 | 双曲空间维度 |
| `lambda_ce` | float | 0.1 | 蕴含损失权重 |
| `lambda_cc` | float | 0.1 | 对比损失权重 |
| `temperature` | float | 0.07 | 对比学习温度 |
| `curvature` | float | 1.0 | 双曲空间曲率 |

#### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_epoch` | int | 100 | 最大训练轮数 |
| `lp_learning_rate` | float | 1e-4 | 学习率 |
| `dropout` | float | 0.6 | Dropout比率 |
| `alpha` | float | 0.2 | LeakyReLU参数 |
| `patience` | int | 50 | 早停patience |

#### 架构参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_dims` | int | 224 | 隐层维度 |
| `struct_cmt_num` | int | 300 | Locality数量 |
| `fnc_cmt_num` | int | 30 | Region数量 |
| `node_dims` | int | 128 | 节点嵌入维度 |

### 创建自定义配置

```json
{
    "hyperbolic_dim": 256,
    "lambda_ce": 0.15,
    "lambda_cc": 0.12,
    "temperature": 0.08,
    "lp_learning_rate": 8e-5,
    "max_epoch": 150,
    "dropout": 0.6,
    "hidden_dims": 256,
    "struct_cmt_num": 300,
    "fnc_cmt_num": 30
}
```

使用：
```bash
python run_hrnr_hyperbolic.py --config_file my_config.json
```

## 📈 实验工作流

### 推荐的完整实验流程

```bash
# 1. 初步测试（快速验证）
./quick_start.sh --mode single --dataset xa

# 2. 超参数粗搜索（30-50次随机搜索）
python hyperparameter_tuning.py \
    --method random \
    --max_trials 50 \
    --dataset xa

# 3. 超参数细搜索（在最佳区域）
# 根据粗搜索结果创建custom_search_space.json
python hyperparameter_tuning.py \
    --search_space_file custom_search_space.json \
    --method grid \
    --max_trials 30

# 4. 使用最佳配置运行多随机种子
python run_hrnr_hyperbolic.py \
    --mode multi_seed \
    --num_runs 5 \
    --config_file best_config.json

# 5. 消融实验
./quick_start.sh --mode ablation --dataset xa

# 6. 模型对比
./quick_start.sh --mode comparison --dataset xa

# 7. 生成可视化报告
python -c "
from visualization_tools import ExperimentVisualizer
vis = ExperimentVisualizer()
vis.plot_all_results()
"
```

## 🔍 结果分析

### 结果文件位置

- **模型**: `veccity/cache/{exp_id}/model_cache/`
- **嵌入**: `veccity/cache/{exp_id}/evaluate_cache/road_embedding_*.npy`
- **日志**: `veccity/log/{exp_id}-*.log`
- **实验结果**: `experiments/results/*.json`
- **可视化**: `experiments/figures/*.png`

### 评估指标

| 指标 | 说明 | 优化目标 |
|------|------|----------|
| **AUC** | ROC曲线下面积 | 越高越好 |
| **F1** | F1分数 | 越高越好 |
| **Precision** | 精确率 | 越高越好 |
| **Recall** | 召回率 | 越高越好 |

### 统计分析

对于多随机种子实验，报告包含：
- 均值（Mean）
- 标准差（Std）
- 最大值（Max）
- 最小值（Min）
- 95%置信区间

## ⚡ 性能优化建议

### 计算资源优化

1. **减少计算量**:
   ```python
   # 在蕴含损失和对比损失中减少采样
   # 修改HRNR_Hyperbolic.py中的采样参数
   sample_size = min(5, len(segments_idx))  # 从10改为5
   sample_edges = min(500, num_edges)       # 从1000改为500
   ```

2. **GPU加速**:
   ```bash
   # 确保使用GPU
   --gpu True --gpu_id 0
   ```

3. **批次大小调整**:
   ```json
   {
     "batch_size": 32  // 从64减小到32可减少显存占用
   }
   ```

### 训练效率

1. **学习率调度**: 启用warmup和cosine annealing
2. **早停**: 设置合适的patience避免过度训练
3. **梯度累积**: 大模型时使用
4. **混合精度**: 使用torch.cuda.amp（需要PyTorch >= 1.6）

## 🐛 常见问题

### Q1: 训练太慢

**A**:
- 使用GPU: `--gpu True`
- 减小batch_size
- 减少采样数量
- 使用更小的hyperbolic_dim

### Q2: 显存不足

**A**:
- 减小batch_size
- 减小hyperbolic_dim
- 使用梯度累积: `"grad_accmu_steps": 2`

### Q3: NaN损失

**A**:
- 检查学习率（可能太大）
- 检查数值稳定性（eps参数）
- 使用梯度裁剪: `"clip_grad_norm": True`

### Q4: 超参数搜索失败

**A**:
- 检查搜索空间是否合理
- 减少max_trials
- 使用随机搜索代替网格搜索

## 📚 扩展开发

### 添加新的实验模式

在`run_hrnr_hyperbolic.py`中添加：

```python
def run_custom_experiment(args):
    """自定义实验逻辑"""
    # 实现你的实验
    pass

# 在main()中注册
if args.mode == 'custom':
    run_custom_experiment(args)
```

### 添加新的可视化

在`visualization_tools.py`中添加：

```python
def plot_custom_analysis(self, data, save_path=None):
    """自定义可视化"""
    # 实现你的可视化
    pass
```

### 添加新的优化算法

在`hyperparameter_tuning.py`中扩展：

```python
def run_custom_optimization(self):
    """自定义优化算法"""
    # 实现你的优化逻辑
    pass
```

## 📖 参考资料

### 相关文档

- **模型文档**: `HRNR_HYPERBOLIC_README.md`
- **实验指南**: `experiments/README.md`
- **VecCity文档**: `VecCity-main/README.md`

### 论文引用

```bibtex
@inproceedings{hrnr_hyperbolic2025,
  title={HRNR with Hyperbolic Embeddings for Hierarchical Road Network Representation},
  author={Your Name},
  booktitle={Conference},
  year={2025}
}
```

## 💡 最佳实践

1. **实验前**:
   - 检查数据完整性
   - 验证配置合理性
   - 预估计算资源

2. **实验中**:
   - 监控训练曲线
   - 及时保存checkpoints
   - 记录重要观察

3. **实验后**:
   - 保存完整配置
   - 生成可视化报告
   - 备份重要结果

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议：

- GitHub Issues: https://github.com/jefferyou/hperroad/issues
- Pull Requests: 欢迎提交改进

## 📞 联系方式

- Email: your.email@example.com
- GitHub: @yourusername

---

**祝实验顺利！**
