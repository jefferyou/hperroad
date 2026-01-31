# HRNR_Hyperbolic 实验指南

完整的实验流程和工具使用说明。

## 目录结构

```
experiments/
├── README.md                      # 本文件
├── run_hrnr_hyperbolic.py        # 主实验运行脚本
├── hyperparameter_tuning.py      # 超参数优化脚本
├── visualization_tools.py         # 可视化工具
├── quick_start.sh                # 快速启动脚本
├── results/                      # 实验结果存储目录
└── figures/                      # 可视化图片存储目录
```

## 快速开始

### 1. 单次实验

运行一次完整的训练和评估：

```bash
cd experiments
python run_hrnr_hyperbolic.py \
    --dataset xa \
    --seed 0 \
    --gpu True \
    --gpu_id 0
```

### 2. 多随机种子实验

运行多次实验以评估稳定性：

```bash
python run_hrnr_hyperbolic.py \
    --mode multi_seed \
    --num_runs 5 \
    --dataset xa
```

### 3. 消融实验

测试不同组件的贡献：

```bash
python run_hrnr_hyperbolic.py \
    --mode ablation \
    --dataset xa
```

这会自动运行以下配置：
- 完整模型（蕴含损失 + 对比损失）
- 无蕴含损失
- 无对比损失
- 仅结构损失

### 4. 模型对比

对比HRNR和HRNR_Hyperbolic：

```bash
python run_hrnr_hyperbolic.py \
    --mode comparison \
    --dataset xa
```

### 5. 超参数优化

#### 随机搜索（推荐）

```bash
python hyperparameter_tuning.py \
    --method random \
    --max_trials 50 \
    --dataset xa \
    --metric auc
```

#### 网格搜索

```bash
python hyperparameter_tuning.py \
    --method grid \
    --max_trials 100 \
    --dataset xa
```

#### 使用自定义搜索空间

创建搜索空间配置文件 `search_space.json`：

```json
{
    "hyperbolic_dim": [128, 224, 256],
    "lambda_ce": [0.05, 0.1, 0.15, 0.2],
    "lambda_cc": [0.05, 0.1, 0.15, 0.2],
    "temperature": [0.05, 0.07, 0.1],
    "lp_learning_rate": [5e-5, 1e-4, 2e-4]
}
```

然后运行：

```bash
python hyperparameter_tuning.py \
    --search_space_file search_space.json \
    --method random \
    --max_trials 30
```

## 命令行参数详解

### run_hrnr_hyperbolic.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | road_representation | 任务类型 |
| `--model` | str | HRNR_Hyperbolic | 模型名称 |
| `--dataset` | str | xa | 数据集名称 |
| `--mode` | str | single | 实验模式（single/multi_seed/ablation/comparison） |
| `--gpu` | bool | True | 是否使用GPU |
| `--gpu_id` | int | 0 | GPU编号 |
| `--seed` | int | 0 | 随机种子 |
| `--num_runs` | int | 5 | multi_seed模式下的运行次数 |
| `--hyperbolic_dim` | int | 224 | 双曲空间维度 |
| `--lambda_ce` | float | 0.1 | 蕴含损失权重 |
| `--lambda_cc` | float | 0.1 | 对比损失权重 |
| `--temperature` | float | 0.07 | 对比学习温度 |
| `--learning_rate` | float | 1e-4 | 学习率 |
| `--max_epoch` | int | 100 | 最大训练轮数 |

### hyperparameter_tuning.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--method` | str | random | 搜索方法（grid/random/bayesian） |
| `--max_trials` | int | 50 | 最大尝试次数 |
| `--metric` | str | auc | 优化目标指标 |
| `--mode` | str | max | 优化模式（max/min） |
| `--search_space_file` | str | None | 搜索空间配置文件 |

## 结果分析

### 查看实验结果

实验结果保存在 `results/` 目录下，包括：

1. **单次实验结果**: JSON格式，包含所有评估指标
2. **多随机种子汇总**: 包含每次运行的结果和统计信息
3. **消融实验结果**: 各配置的对比
4. **超参数优化结果**: 所有试验的记录和最佳配置

### 可视化结果

使用可视化工具分析结果：

```python
from visualization_tools import ExperimentVisualizer

visualizer = ExperimentVisualizer()

# 训练曲线
visualizer.plot_training_curves('path/to/log.log')

# 超参数重要性
visualizer.plot_hyperparameter_importance('results/hypertuning_*.json')

# 消融实验
visualizer.plot_ablation_study('results/*_ablation_study.json')

# 模型对比
visualizer.plot_model_comparison('results/hrnr_comparison_*.json')

# 嵌入可视化
visualizer.plot_embedding_pca('path/to/embeddings.npy')
```

## 实验最佳实践

### 1. 超参数调优流程

推荐的调优顺序：

1. **粗搜索**：使用随机搜索（30-50次），大范围探索
   ```bash
   python hyperparameter_tuning.py --method random --max_trials 50
   ```

2. **细搜索**：在最佳区域附近使用网格搜索
   - 根据粗搜索结果缩小搜索空间
   - 创建精细化的search_space.json
   - 运行网格搜索

3. **验证**：使用最佳配置运行多随机种子实验
   ```bash
   python run_hrnr_hyperbolic.py --mode multi_seed --num_runs 5
   ```

### 2. 重要超参数建议

基于理论和经验的建议：

| 超参数 | 推荐范围 | 说明 |
|--------|----------|------|
| `hyperbolic_dim` | [128, 256] | 与hidden_dims保持一致或接近 |
| `lambda_ce` | [0.05, 0.2] | 蕴含损失权重，过大会过度约束 |
| `lambda_cc` | [0.05, 0.2] | 对比损失权重，根据任务调整 |
| `temperature` | [0.05, 0.1] | 温度越低对比越严格 |
| `learning_rate` | [5e-5, 2e-4] | 双曲操作需要较小学习率 |
| `dropout` | [0.5, 0.7] | 防止过拟合 |

### 3. 消融实验建议

测试各组件的贡献：

1. **基线**: 完整模型
2. **-蕴含**: lambda_ce=0（测试蕴含锥的作用）
3. **-对比**: lambda_cc=0（测试对比学习的作用）
4. **-双曲**: 使用原始HRNR（测试双曲空间的作用）

### 4. 数据集选择

VecCity支持多个数据集，推荐用于测试：

- **xa**: 西安路网（中等规模，适合快速实验）
- **bj**: 北京路网（大规模，测试scalability）
- **porto**: Porto出租车数据（真实轨迹）

## 常见问题

### Q1: 训练很慢怎么办？

**A**: 双曲操作比欧氏操作慢，可以：
1. 减小batch_size
2. 使用GPU加速
3. 减少采样数量（在蕴含损失和对比损失中）
4. 使用更小的模型维度

### Q2: 显存不足怎么办？

**A**:
1. 减小batch_size
2. 减小hyperbolic_dim
3. 使用梯度累积（grad_accmu_steps）
4. 减少层次数量

### Q3: 如何选择最佳配置？

**A**:
1. 先运行超参数搜索
2. 选择top-3配置
3. 对每个配置运行5次多随机种子实验
4. 选择均值最高且方差最小的配置

### Q4: 如何复现论文结果？

**A**:
1. 使用固定随机种子
2. 使用论文中的超参数配置
3. 运行多次求平均（建议5次）
4. 确保数据预处理一致

## 高级用法

### 1. 自定义实验配置

创建配置文件 `custom_config.json`：

```json
{
    "hyperbolic_dim": 256,
    "lambda_ce": 0.15,
    "lambda_cc": 0.12,
    "temperature": 0.08,
    "lp_learning_rate": 8e-5,
    "max_epoch": 150,
    "dropout": 0.6,
    "alpha": 0.2
}
```

使用自定义配置：

```bash
python run_hrnr_hyperbolic.py \
    --config_file custom_config.json \
    --dataset xa
```

### 2. 批量实验

创建批量运行脚本 `batch_experiments.sh`：

```bash
#!/bin/bash

datasets=("xa" "bj" "porto")
seeds=(0 1 2 3 4)

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Running experiment: dataset=$dataset, seed=$seed"
        python run_hrnr_hyperbolic.py \
            --dataset $dataset \
            --seed $seed \
            --exp_id "batch_${dataset}_s${seed}"
    done
done
```

### 3. 分布式超参数搜索

在多GPU上并行搜索：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python hyperparameter_tuning.py \
    --max_trials 25 --seed 0 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python hyperparameter_tuning.py \
    --max_trials 25 --seed 1 &

wait
```

## 性能基准

在xa数据集上的参考性能（单次运行）：

| 模型 | AUC | F1 | Precision | Recall |
|------|-----|-------|-----------|--------|
| HRNR (baseline) | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| HRNR_Hyperbolic | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| 提升 | +X.X% | +X.X% | +X.X% | +X.X% |

*注: 实际结果需要通过实验获得*

## 引用

如果使用本实验框架，请引用：

```bibtex
@inproceedings{hrnr_hyperbolic2025,
  title={HRNR with Hyperbolic Embeddings for Hierarchical Road Network Representation},
  author={Your Name},
  year={2025}
}
```

## 联系方式

如有问题或建议，请联系：
- Email: your.email@example.com
- GitHub Issues: https://github.com/yourusername/hperroad/issues

---

**祝实验顺利！**
