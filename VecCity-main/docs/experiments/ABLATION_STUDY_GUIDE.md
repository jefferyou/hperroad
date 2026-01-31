# HRNR_Hyperbolic 消融实验完整指南

## 📋 概述

本指南介绍如何运行HRNR_Hyperbolic的完整消融实验，包括：
- 5个不同的配置（baseline + 4个消融版本）
- 断点续传支持
- GPU加速
- 自动结果分析和可视化

## 🔬 消融实验配置

### 1. **Baseline HRNR** (baseline_hrnr)
- 原始HRNR模型（欧氏空间）
- 仅使用交叉熵损失
- 作为对比基准

### 2. **Full HRNR_Hyperbolic** (full_model)
- 完整的双曲空间模型
- 包含蕴含损失 (λ_CE = 0.1)
- 包含对比损失 (λ_CC = 0.1)
- **最佳性能配置**

### 3. **HRNR_Hyp w/o Entailment** (no_entailment)
- 双曲空间 + 对比损失
- λ_CE = 0
- λ_CC = 0.1
- **测试蕴含锥的贡献**

### 4. **HRNR_Hyp w/o Contrastive** (no_contrastive)
- 双曲空间 + 蕴含损失
- λ_CE = 0.1
- λ_CC = 0
- **测试对比学习的贡献**

### 5. **HRNR_Hyp w/o Auxiliary** (no_auxiliary)
- 仅双曲空间表示
- λ_CE = 0
- λ_CC = 0
- **测试双曲空间本身的贡献**

## 🚀 快速开始

### 方式1：使用启动脚本（推荐）

```bash
cd /home/user/hperroad/experiments

# 基本用法（使用默认参数）
./run_ablation.sh

# 指定数据集和种子
./run_ablation.sh --dataset xa --seed 0

# 使用特定GPU
./run_ablation.sh --gpu-id 1

# 从中断处继续
./run_ablation.sh --resume

# 跳过baseline（只运行hyperbolic变体）
./run_ablation.sh --skip-baseline
```

### 方式2：直接使用Python脚本

```bash
cd /home/user/hperroad/experiments

# 基本运行
python run_complete_ablation.py --dataset xa --seed 0 --gpu true --gpu_id 0

# 断点续传
python run_complete_ablation.py --dataset xa --seed 0 --resume

# 跳过baseline
python run_complete_ablation.py --dataset xa --seed 0 --skip_baseline
```

## ⚙️ 参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | xa | 数据集名称 |
| `--seed` | 0 | 随机种子 |
| `--task` | segment | 任务类型 |

### GPU配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu` | true | 是否使用GPU |
| `--gpu_id` | 0 | GPU设备ID |

**注意**：GPU配置会自动传递到下游任务（STS, TTE, TSI）

### 双曲空间参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hyperbolic_dim` | 224 | 双曲空间维度 |
| `--lambda_ce` | 0.1 | 蕴含损失权重（完整模型） |
| `--lambda_cc` | 0.1 | 对比损失权重（完整模型） |
| `--temperature` | 0.07 | 对比学习温度 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--max_epoch` | 100 | 最大训练轮数 |

### 消融实验特定

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--resume` | false | 从进度文件继续 |
| `--skip_baseline` | false | 跳过baseline HRNR |

## 📁 文件结构

实验运行后会生成以下文件：

```
experiments/results/ablation/
├── ablation_progress_xa_seed0.json      # 进度文件（用于断点续传）
├── ablation_results_xa_seed0.json       # 完整结果文件
├── ablation_summary_xa_seed0.txt        # 文本总结
├── ablation_comparison.csv              # CSV对比表格
├── ablation_comparison.md               # Markdown报告
├── component_analysis.txt               # 组件贡献分析
└── ablation_*.png                       # 性能对比图（如果有matplotlib）
```

## 🔄 断点续传

### 自动保存进度

实验会自动保存进度到：
```
experiments/results/ablation/ablation_progress_{dataset}_seed{seed}.json
```

进度文件包含：
- ✅ 已完成的实验ID
- ❌ 失败的实验及错误信息
- 🔄 当前正在运行的实验

### 从中断处继续

如果实验中断（如网络断开、服务器重启），可以使用：

```bash
# 方式1：使用启动脚本
./run_ablation.sh --resume

# 方式2：使用Python脚本
python run_complete_ablation.py --dataset xa --seed 0 --resume
```

系统会：
1. 读取进度文件
2. 跳过已完成的实验
3. 从下一个未完成的实验继续

### 查看进度

```bash
# 查看进度文件
cat experiments/results/ablation/ablation_progress_xa_seed0.json
```

## 📊 结果分析

### 自动分析

运行完成后，脚本会自动调用分析工具：

```bash
python analyze_ablation_results.py --results_file experiments/results/ablation/ablation_results_xa_seed0.json
```

### 手动分析

如果需要重新分析已有结果：

```bash
cd /home/user/hperroad/experiments

python analyze_ablation_results.py \
    --results_file results/ablation/ablation_results_xa_seed0.json
```

### 生成的分析内容

1. **对比表格** (`ablation_comparison.csv`)
   - 所有配置的性能指标
   - 可用Excel打开

2. **Markdown报告** (`ablation_comparison.md`)
   - 包含详细的配置说明
   - 适合直接查看或插入文档

3. **组件贡献分析** (`component_analysis.txt`)
   - 每个组件的性能提升
   - baseline vs full model的对比

4. **可视化图表** (`ablation_*.png`)
   - 各配置的性能柱状图
   - 需要matplotlib

## 🎯 下游任务评估

每个配置都会在以下任务上评估：

1. **STS** - Similarity Search（相似度搜索）
2. **TTE** - Travel Time Estimation（行程时间估计）
3. **TSI** - Traffic Speed Inference（速度推断）

### GPU加速配置

所有下游任务会自动使用GPU（如果启用）：

```python
other_args = {
    'gpu': True,      # 自动传递
    'gpu_id': 0,      # 自动传递
    ...
}
```

## 💡 使用技巧

### 1. 多种子实验

运行多个不同种子以获得更可靠的结果：

```bash
for seed in 0 1 2 3 4; do
    ./run_ablation.sh --dataset xa --seed $seed
done
```

### 2. 只运行hyperbolic变体

如果已经有baseline结果，可以跳过：

```bash
./run_ablation.sh --skip-baseline
```

### 3. 自定义超参数

修改Python脚本中的默认参数，或直接传递：

```bash
python run_complete_ablation.py \
    --dataset xa \
    --seed 0 \
    --lambda_ce 0.2 \
    --lambda_cc 0.15 \
    --temperature 0.05
```

### 4. 检查特定实验

如果某个配置失败，可以查看错误信息：

```bash
cat experiments/results/ablation/ablation_progress_xa_seed0.json | grep -A 5 "failed"
```

## 🐛 常见问题

### Q1: GPU内存不足

**症状**：CUDA out of memory

**解决方案**：
1. 减小batch_size（在配置文件中）
2. 使用更大内存的GPU
3. 或临时禁用GPU：`./run_ablation.sh --no-gpu`

### Q2: 某个实验一直失败

**解决方案**：
1. 查看进度文件中的错误信息
2. 手动运行该配置进行调试：
```bash
python run_complete_ablation.py --dataset xa --seed 0
# 修改代码跳过其他配置，只运行失败的
```

### Q3: 如何重新运行某个配置

**解决方案**：
1. 编辑进度文件，从`completed`列表中移除该配置ID
2. 使用`--resume`重新运行

### Q4: 下游任务没有使用GPU

**检查**：
1. 查看日志中的设备信息
2. 确认GPU参数正确传递：
```python
print(other_args)  # 添加调试输出
```

## 📈 预期结果

基于理论分析，预期的性能排序：

```
Full Model > w/o Contrastive ≈ w/o Entailment > w/o Auxiliary > Baseline
```

各组件的贡献：
- **双曲空间**：~5-10%性能提升
- **蕴含损失**：~2-5%额外提升
- **对比损失**：~2-5%额外提升

## 📝 实验记录建议

为每次实验创建记录：

```bash
# 创建实验日志
./run_ablation.sh --dataset xa --seed 0 2>&1 | tee experiment_log_$(date +%Y%m%d_%H%M%S).txt
```

## 🔗 相关文档

- [HRNR_HYPERBOLIC_README.md](../HRNR_HYPERBOLIC_README.md) - 模型详细说明
- [EXPERIMENT_FRAMEWORK_README.md](../EXPERIMENT_FRAMEWORK_README.md) - 实验框架
- [experiments/README.md](README.md) - 实验总览

## 📧 问题反馈

如有问题，请检查：
1. 进度文件中的错误信息
2. VecCity日志输出
3. GPU可用性（`nvidia-smi`）

---

**创建日期**: 2025-01-26
**版本**: v1.0
**作者**: Claude Code
