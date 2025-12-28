# 快速测试指南

## 所有问题已修复 ✅

✅ 路径问题已修复
✅ 参数传递问题已修复

现在脚本可以从**任何目录**正常运行。

## 如何运行实验

### 方法1: 使用Python脚本（推荐）

```bash
# 在Windows上
cd C:\Users\Administrator\Downloads\hperroad-claude-hyperbolic-embeddings-veccity-9Rpvs\experiments

# 运行单次实验
python run_hrnr_hyperbolic.py --dataset xa --seed 0

# 运行超参数优化
python hyperparameter_tuning.py --method random --max_trials 10
```

### 方法2: 使用Shell脚本（Linux/Mac）

```bash
cd /path/to/hperroad/experiments

# 单次实验
./quick_start.sh --mode single --dataset xa

# 多随机种子
./quick_start.sh --mode multi_seed --dataset xa
```

## 修复内容

✅ **工作目录**: 脚本自动切换到VecCity-main目录
✅ **保存路径**: 使用绝对路径保存结果
✅ **跨平台**: Windows和Linux都支持
✅ **结果目录**: 自动创建experiments/results和experiments/figures

## 测试步骤

### 1. 最小测试（快速验证）

```bash
python run_hrnr_hyperbolic.py \
    --dataset xa \
    --seed 0 \
    --max_epoch 2 \
    --gpu True \
    --gpu_id 0
```

预期输出：
```
================================================================================
Running HRNR_Hyperbolic experiment
Dataset: xa
Seed: 0
GPU: True (ID: 0)
================================================================================
Starting training with Hyperbolic Embeddings...
...
```

### 2. 验证数据加载

确保以下数据文件存在：
- `VecCity-main/raw_data/xa/xa.geo`
- `VecCity-main/raw_data/xa/xa.rel`
- `VecCity-main/raw_data/xa/xa.dyna`（可选）

### 3. 检查结果

实验完成后检查：
```bash
# Windows
dir ..\experiments\results

# Linux/Mac
ls ../experiments/results
```

应该看到JSON结果文件。

## 常见问题

### Q: FileNotFoundError: task_config.json

**A**: 这个问题已修复。如果仍出现，请：
```bash
git pull origin claude/hyperbolic-embeddings-veccity-9Rpvs
```

### Q: CUDA out of memory

**A**: 减小batch_size或使用CPU：
```bash
python run_hrnr_hyperbolic.py --gpu False
```

### Q: 找不到数据集

**A**: 检查数据集位置：
```bash
# 应该在这里
VecCity-main/raw_data/{dataset}/
```

### Q: 模块导入错误

**A**: 确保在正确的Python环境：
```bash
# 激活环境
conda activate VecCity

# 检查路径
python -c "import sys; print(sys.path)"
```

## 完整实验示例

```bash
# 激活环境
conda activate VecCity

# 进入实验目录
cd experiments

# 1. 单次快速测试（2个epoch）
python run_hrnr_hyperbolic.py \
    --dataset xa \
    --max_epoch 2 \
    --seed 0

# 2. 完整单次实验（100个epoch）
python run_hrnr_hyperbolic.py \
    --dataset xa \
    --seed 0

# 3. 多随机种子（3次运行）
python run_hrnr_hyperbolic.py \
    --mode multi_seed \
    --num_runs 3 \
    --dataset xa

# 4. 消融实验
python run_hrnr_hyperbolic.py \
    --mode ablation \
    --dataset xa

# 5. 小规模超参数搜索（10次）
python hyperparameter_tuning.py \
    --method random \
    --max_trials 10 \
    --dataset xa
```

## 预期运行时间（GPU）

- 单次测试（2 epoch）: ~5-10分钟
- 单次完整（100 epoch）: ~2-4小时
- 多随机种子（5次）: ~10-20小时
- 消融实验（4个配置）: ~8-16小时
- 超参数搜索（50次）: ~3-5天

## 结果位置

```
hperroad/
├── experiments/
│   ├── results/                    # JSON结果
│   │   ├── hrnr_hyperbolic_xa_multi_seed_summary.json
│   │   ├── hrnr_hyperbolic_xa_ablation_study.json
│   │   └── hypertuning_HRNR_Hyperbolic_xa_*.json
│   └── figures/                    # 可视化图片
│       ├── training_curves.png
│       ├── hyperparameter_importance.png
│       └── ablation_study.png
│
└── VecCity-main/
    ├── veccity/cache/{exp_id}/     # 模型缓存
    │   ├── model_cache/
    │   └── evaluate_cache/
    └── veccity/log/                # 训练日志
        └── {exp_id}-*.log
```

## 下一步

实验运行成功后：

1. **查看日志**:
   ```bash
   # 找到最新的日志
   ls -lt VecCity-main/veccity/log/

   # 查看日志
   tail -f VecCity-main/veccity/log/{exp_id}.log
   ```

2. **分析结果**:
   ```python
   import json
   with open('experiments/results/hrnr_hyperbolic_xa_*.json') as f:
       results = json.load(f)
   print(results)
   ```

3. **可视化**:
   ```python
   from visualization_tools import ExperimentVisualizer
   vis = ExperimentVisualizer()
   vis.plot_training_curves('path/to/log.log')
   ```

## 获取帮助

如果遇到问题：

1. 检查日志文件
2. 查看`EXPERIMENT_FRAMEWORK_README.md`
3. 运行`python example_usage.py`查看示例

---

**祝实验顺利！** 🚀
