# HRNR_Hyperbolic 实验脚本说明

本目录包含用于运行HRNR_Hyperbolic模型多城市下游任务评估的完整实验脚本。

## 📁 文件清单

### 主要脚本

| 文件 | 类型 | 说明 |
|------|------|------|
| `run_hrnr_experiments.py` | Python | **主实验脚本**（推荐），支持自动收集和分析结果 |
| `run_hrnr_experiments.sh` | Bash | 简化版bash脚本，适合快速批量运行 |
| `quick_start.sh` | Bash | **一键启动脚本**，包含环境检查和自动化流程 |
| `analyze_results.py` | Python | 结果分析脚本，支持多种输出格式 |
| `run_downstream_only.py` | Python | 单次实验运行脚本（已存在） |

### 文档

| 文件 | 说明 |
|------|------|
| `EXPERIMENT_GUIDE.md` | **详细实验指南**，包含完整使用说明和故障排除 |
| `EXPERIMENTS_README.md` | 本文件，快速参考 |

## 🚀 快速开始

### 方法1: 一键启动（最简单）

```bash
bash quick_start.sh
```

这个脚本会自动：
- ✓ 检查环境和依赖
- ✓ 验证embeddings文件
- ✓ 运行所有实验
- ✓ 分析结果生成报告

### 方法2: Python脚本（推荐，功能最全）

```bash
# 使用默认配置（5城市×5种子=25个实验）
python run_hrnr_experiments.py --task segment --device gpu

# 自定义配置
python run_hrnr_experiments.py \
    --datasets bj cd prt \
    --seeds 31 42 53 \
    --device gpu
```

### 方法3: Bash脚本（简单直接）

```bash
# 默认配置
./run_hrnr_experiments.sh

# 指定设备
./run_hrnr_experiments.sh gpu 1
```

## 📊 实验配置

默认实验设置（符合论文要求）：

- **数据集**: prt, cd, bj, xa, df (5个城市)
- **随机种子**: 31, 42, 53, 64, 75 (5次运行)
- **Embedding维度**: 128
- **总实验数**: 25 (5城市 × 5种子)
- **预计耗时**: 5-12小时

## 📈 结果分析

实验完成后，使用分析脚本：

```bash
# 自动分析（如果使用quick_start.sh会自动执行）
python analyze_results.py ./experiment_results/20260203_120000

# 生成不同格式的报告
python analyze_results.py ./experiment_results/20260203_120000 --format csv      # CSV表格
python analyze_results.py ./experiment_results/20260203_120000 --format markdown # Markdown
python analyze_results.py ./experiment_results/20260203_120000 --format latex    # LaTeX表格
python analyze_results.py ./experiment_results/20260203_120000 --format json     # JSON数据
```

## 📝 输出文件

### Python脚本自动生成

```
experiment_results/20260203_120000/
├── raw_results_20260203_120000.json          # 原始结果（所有实验）
├── aggregated_results_20260203_120000.json   # 聚合结果（均值±标准差）
├── results_table_20260203_120000.csv         # CSV表格（易于导入Excel）
└── report_20260203_120000.txt                # 文本报告（易读）
```

### Bash脚本生成

```
experiment_results/20260203_120000/
├── experiment_log.txt                         # 总日志
├── result_bj_seed31.log                      # 各实验日志
├── result_bj_seed42.log
└── ...
```

可用`analyze_results.py`分析bash脚本的结果。

## 🔧 常见用法

### 只运行部分数据集

```bash
# Python
python run_hrnr_experiments.py --datasets bj cd --device gpu

# Bash
# 需要修改脚本中的DATASETS变量

# 快速启动
bash quick_start.sh --datasets "bj cd"
```

### 只运行3次（快速测试）

```bash
# Python
python run_hrnr_experiments.py --seeds 31 42 53 --device gpu

# 快速启动
bash quick_start.sh --seeds "31 42 53"
```

### 使用CPU

```bash
# Python
python run_hrnr_experiments.py --device cpu

# Bash
./run_hrnr_experiments.sh cpu

# 快速启动
bash quick_start.sh --device cpu
```

### 手动运行单个实验

```bash
python run_downstream_only.py \
    --task segment \
    --model HRNR_Hyperbolic \
    --dataset bj \
    --seed 31 \
    --output_dim 128 \
    --device gpu \
    --exp_id 1
```

## 📖 详细文档

完整使用说明、故障排除、高级用法请参考：

👉 **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)**

## ⚠️ 注意事项

1. **运行前准备**
   - 确保已训练模型生成embeddings
   - 检查GPU内存（建议16GB+）
   - 预留足够的时间（5-12小时）

2. **内存管理**
   - 如果遇到OOM错误，使用`--device cpu`
   - 或减少并发数据集数量

3. **中断恢复**
   - 如果实验中断，可以手动重新运行失败的部分
   - 或修改脚本跳过已完成的实验

4. **结果验证**
   - 检查日志文件确认实验成功
   - 验证metrics是否成功提取
   - 对比不同运行之间的标准差

## 📞 问题反馈

如遇到问题：

1. 查看 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) 的故障排除章节
2. 检查日志文件：`./experiment_results/*/experiment_log.txt`
3. 使用 `--verbose` 选项查看详细输出

## 🎯 推荐工作流程

```bash
# 1. 首次使用：一键启动（包含环境检查）
bash quick_start.sh

# 2. 后续使用：直接运行Python脚本
python run_hrnr_experiments.py --task segment --device gpu

# 3. 分析结果
python analyze_results.py ./experiment_results/20260203_120000 --format csv
python analyze_results.py ./experiment_results/20260203_120000 --format markdown

# 4. 查看报告
cat ./experiment_results/20260203_120000/report_*.txt
```

## 📚 脚本对比

| 特性 | quick_start.sh | run_hrnr_experiments.py | run_hrnr_experiments.sh |
|------|----------------|------------------------|------------------------|
| 难度 | ⭐ 最简单 | ⭐⭐ 简单 | ⭐⭐ 简单 |
| 功能 | 自动化全流程 | 功能最全 | 基础批量运行 |
| 环境检查 | ✓ | ✗ | ✗ |
| 自动分析 | ✓ | ✓ | ✗ (需手动) |
| 结果格式 | 多种 | 多种 | 日志文件 |
| 可定制性 | 中 | 高 | 中 |
| 推荐场景 | 首次使用 | 日常使用 | 简单批处理 |

## 💡 提示

- 第一次运行建议使用 `quick_start.sh` 确保环境正确
- 日常实验使用 `run_hrnr_experiments.py` 获得最佳体验
- 需要简单快速批量运行时使用 `run_hrnr_experiments.sh`
- 所有脚本都支持中断后重新运行

---

**最后更新**: 2026-02-03
**维护者**: [你的名字]
