# HRNR_Hyperbolic 消融实验系统

## 📚 快速导航

- [消融实验完整指南](ABLATION_STUDY_GUIDE.md) - 详细使用文档
- [主实验脚本](run_complete_ablation.py) - Python脚本
- [启动脚本](run_ablation.sh) - Bash快速启动
- [结果分析脚本](analyze_ablation_results.py) - 自动分析工具

## 🎯 功能特性

### ✅ 完整的消融实验配置

| ID | 名称 | 描述 |
|----|------|------|
| `baseline_hrnr` | Baseline HRNR | 原始欧氏空间HRNR |
| `full_model` | Full HRNR_Hyperbolic | 完整双曲模型（蕴含+对比） |
| `no_entailment` | w/o Entailment | 无蕴含损失 |
| `no_contrastive` | w/o Contrastive | 无对比损失 |
| `no_auxiliary` | w/o Auxiliary | 仅双曲空间 |

### ✅ 断点续传支持

- 自动保存实验进度
- 从中断位置继续
- 跳过已完成的实验

### ✅ GPU加速

- 自动传递GPU配置到下游任务
- 支持多GPU选择
- CPU备用模式

### ✅ 自动结果分析

- CSV/Markdown对比表格
- 组件贡献分析
- 性能可视化图表

## 🚀 快速开始

### 1. 环境准备

```bash
# 1. 进入VecCity环境（假设已安装）
cd /home/user/hperroad/VecCity-main

# 2. 确保VecCity依赖已安装
pip install -r requirements.txt  # 如果有的话

# 3. 检查环境
cd ../experiments
python test_ablation_setup.py
```

### 2. 运行消融实验

```bash
cd /home/user/hperroad/experiments

# 方式1：使用启动脚本（推荐）
./run_ablation.sh --dataset xa --seed 0

# 方式2：直接Python
python run_complete_ablation.py --dataset xa --seed 0 --gpu true --gpu_id 0

# 方式3：断点续传
./run_ablation.sh --resume
```

### 3. 查看结果

```bash
# 查看结果目录
ls -la results/ablation/

# 查看Markdown报告
cat results/ablation/ablation_comparison.md

# 查看CSV
cat results/ablation/ablation_comparison.csv
```

## 📊 实验配置详解

### 消融实验矩阵

| 配置 | 双曲空间 | 蕴含损失(λ_CE) | 对比损失(λ_CC) | 目的 |
|------|---------|---------------|---------------|------|
| Baseline | ❌ | ❌ | ❌ | 性能基准 |
| Full Model | ✅ | 0.1 | 0.1 | 最佳性能 |
| w/o Entailment | ✅ | 0.0 | 0.1 | 测试蕴含锥贡献 |
| w/o Contrastive | ✅ | 0.1 | 0.0 | 测试对比学习贡献 |
| w/o Auxiliary | ✅ | 0.0 | 0.0 | 测试双曲空间本身 |

### 评估任务

每个配置在以下3个任务上评估：

1. **STS** (Similarity Search) - 相似度搜索
2. **TTE** (Travel Time Estimation) - 行程时间估计
3. **TSI** (Traffic Speed Inference) - 速度推断

## 📁 文件说明

### 脚本文件

```
experiments/
├── run_complete_ablation.py        # 主消融实验脚本
├── analyze_ablation_results.py     # 结果分析脚本
├── run_ablation.sh                 # Bash启动脚本
├── test_ablation_setup.py          # 环境测试脚本
├── ABLATION_STUDY_GUIDE.md         # 详细使用指南
└── README_ABLATION.md              # 本文件
```

### 结果文件

```
experiments/results/ablation/
├── ablation_progress_{dataset}_seed{seed}.json      # 进度文件
├── ablation_results_{dataset}_seed{seed}.json       # 完整结果
├── ablation_summary_{dataset}_seed{seed}.txt        # 文本总结
├── ablation_comparison.csv                          # CSV表格
├── ablation_comparison.md                           # Markdown报告
├── component_analysis.txt                           # 组件分析
└── ablation_*.png                                   # 可视化图表
```

## 🔧 常用命令

### 基本运行

```bash
# 默认配置（xa数据集，seed=0，GPU=0）
./run_ablation.sh

# 指定数据集和种子
./run_ablation.sh --dataset xa --seed 0

# 使用CPU
./run_ablation.sh --no-gpu

# 使用特定GPU
./run_ablation.sh --gpu-id 1
```

### 断点续传

```bash
# 从中断处继续
./run_ablation.sh --resume

# 查看进度
cat results/ablation/ablation_progress_xa_seed0.json
```

### 跳过实验

```bash
# 跳过baseline（只运行hyperbolic变体）
./run_ablation.sh --skip-baseline
```

### 多种子实验

```bash
# 运行5个不同种子
for seed in 0 1 2 3 4; do
    ./run_ablation.sh --dataset xa --seed $seed
done
```

### 结果分析

```bash
# 重新分析结果
python analyze_ablation_results.py \
    --results_file results/ablation/ablation_results_xa_seed0.json

# 查看文本总结
cat results/ablation/ablation_summary_xa_seed0.txt

# 查看组件贡献
cat results/ablation/component_analysis.txt
```

## 📈 预期结果

### 性能排序（理论预期）

```
Full Model > w/o Contrastive ≈ w/o Entailment > w/o Auxiliary > Baseline
```

### 各组件贡献（估计）

- **双曲空间** (w/o Auxiliary vs Baseline): ~5-10%
- **蕴含损失** (Full vs w/o Entailment): ~2-5%
- **对比损失** (Full vs w/o Contrastive): ~2-5%
- **总提升** (Full vs Baseline): ~10-20%

## 🐛 故障排除

### Q1: 环境测试失败

```bash
# 运行环境测试
python test_ablation_setup.py

# 查看详细错误信息
# 按照提示安装缺失的依赖
```

### Q2: GPU内存不足

```bash
# 临时使用CPU
./run_ablation.sh --no-gpu

# 或减小batch size（修改配置文件）
```

### Q3: 实验中断了

```bash
# 直接使用 --resume 继续
./run_ablation.sh --resume

# 系统会自动跳过已完成的实验
```

### Q4: 某个配置一直失败

```bash
# 1. 查看进度文件中的错误
cat results/ablation/ablation_progress_xa_seed0.json

# 2. 手动移除失败标记，重试
# 编辑progress文件，从failed列表中删除该配置

# 3. 使用--resume重新运行
./run_ablation.sh --resume
```

## 💡 最佳实践

### 1. 运行前检查

```bash
# 检查GPU状态
nvidia-smi

# 检查磁盘空间
df -h

# 测试环境
python test_ablation_setup.py
```

### 2. 实验记录

```bash
# 记录实验日志
./run_ablation.sh --dataset xa --seed 0 2>&1 | \
    tee experiment_log_$(date +%Y%m%d_%H%M%S).txt
```

### 3. 多次运行取平均

```bash
# 运行多个种子
for seed in 0 1 2 3 4; do
    ./run_ablation.sh --dataset xa --seed $seed
done

# 然后手动计算平均值和标准差
```

### 4. 分批运行

```bash
# 先运行baseline
python run_complete_ablation.py --dataset xa --seed 0 --skip_baseline false

# 然后运行hyperbolic变体
./run_ablation.sh --skip-baseline --resume
```

## 📊 结果解读

### 查看对比表格

```bash
# Markdown格式（推荐）
cat results/ablation/ablation_comparison.md

# CSV格式（可用Excel打开）
cat results/ablation/ablation_comparison.csv
```

### 查看组件贡献

```bash
cat results/ablation/component_analysis.txt
```

示例输出：
```
================================================================================
COMPONENT CONTRIBUTION ANALYSIS
================================================================================

Metric: AUC
--------------------------------------------------------------------------------
Baseline (HRNR): 0.8234
Full Model (HRNR_Hyperbolic): 0.9012
Improvement: 9.45%

Ablation variants:
  - HRNR_Hyp w/o Entailment: 0.8756
  - HRNR_Hyp w/o Contrastive: 0.8821
  - HRNR_Hyp w/o Auxiliary: 0.8456
```

### 可视化图表

如果安装了matplotlib，会生成柱状图：

```bash
ls results/ablation/*.png
```

## 📖 详细文档

更多详细信息，请参阅：

- **[ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md)** - 完整使用指南
  - 详细的参数说明
  - 断点续传机制
  - 常见问题解答
  - 预期结果分析

- **[../HRNR_HYPERBOLIC_README.md](../HRNR_HYPERBOLIC_README.md)** - 模型文档
  - 模型架构详解
  - 理论背景
  - 技术细节

- **[../EXPERIMENT_FRAMEWORK_README.md](../EXPERIMENT_FRAMEWORK_README.md)** - 实验框架
  - 整体实验设计
  - 基线对比方案

## 🔗 相关链接

- VecCity: https://github.com/LibCity/VecCity
- HyCoCLIP: Hyperbolic Compositional Learning
- HRNR: Hierarchical Road Network Representation

## 📝 更新日志

### v1.0 (2025-01-26)
- ✅ 完整的消融实验框架
- ✅ 5个实验配置
- ✅ 断点续传支持
- ✅ GPU加速
- ✅ 自动结果分析
- ✅ 详细文档

---

**需要帮助？**

1. 首先运行环境测试：`python test_ablation_setup.py`
2. 查看详细指南：`ABLATION_STUDY_GUIDE.md`
3. 检查进度文件：`results/ablation/ablation_progress_*.json`

**快速开始：**
```bash
./run_ablation.sh
```

---

**创建日期**: 2025-01-26
**版本**: v1.0
**作者**: Claude Code
