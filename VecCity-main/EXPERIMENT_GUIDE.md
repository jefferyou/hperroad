# HRNR_Hyperbolic 多城市实验指南

本指南说明如何运行HRNR_Hyperbolic模型在5个城市数据集上的下游任务评估实验。

## 实验设置

根据论文实验要求：

- **数据集**: prt, cd, bj, xa, df (5个城市)
- **Embedding维度**: 128
- **运行次数**: 每个数据集5次（不同随机种子）
- **随机种子**: 31, 42, 53, 64, 75
- **报告格式**: 均值 ± 标准差

## 前置条件

### 1. 确保已有训练好的Embeddings

在运行下游任务评估前，需要先训练模型生成embeddings：

```bash
# 对每个城市数据集训练HRNR_Hyperbolic模型
python run_model.py --task segment --model HRNR_Hyperbolic --dataset prt --device gpu
python run_model.py --task segment --model HRNR_Hyperbolic --dataset cd --device gpu
python run_model.py --task segment --model HRNR_Hyperbolic --dataset bj --device gpu
python run_model.py --task segment --model HRNR_Hyperbolic --dataset xa --device gpu
python run_model.py --task segment --model HRNR_Hyperbolic --dataset df --device gpu
```

训练完成后，embeddings将保存在：
```
./veccity/cache/{exp_id}/evaluate_cache/road_embedding_HRNR_Hyperbolic_{dataset}_128.npy
```

### 2. 验证Embeddings文件

```bash
# 检查embeddings是否存在
ls -lh ./veccity/cache/1/evaluate_cache/road_embedding_HRNR_Hyperbolic_*_128.npy
```

## 运行实验

### 方法1: Python脚本（推荐）

Python脚本提供更好的结果收集和分析功能：

```bash
# 运行所有城市的实验（使用默认配置）
python run_hrnr_experiments.py --task segment --device gpu

# 自定义配置
python run_hrnr_experiments.py \
    --task segment \
    --model HRNR_Hyperbolic \
    --datasets prt cd bj xa df \
    --seeds 31 42 53 64 75 \
    --output_dim 128 \
    --device gpu \
    --exp_id 1 \
    --output_dir ./experiment_results
```

**参数说明**:
- `--task`: 任务类型（segment, region, poi）
- `--model`: 模型名称（默认HRNR_Hyperbolic）
- `--datasets`: 要评估的数据集列表
- `--seeds`: 随机种子列表（用于多次运行）
- `--output_dim`: Embedding维度（默认128）
- `--device`: 设备（cpu, gpu）
- `--exp_id`: 实验ID（用于查找embeddings）
- `--output_dir`: 结果输出目录

### 方法2: Bash脚本

更简单的bash脚本版本：

```bash
# 给脚本添加执行权限
chmod +x run_hrnr_experiments.sh

# 运行实验（使用默认GPU）
./run_hrnr_experiments.sh

# 指定设备和实验ID
./run_hrnr_experiments.sh gpu 1
```

### 方法3: 手动运行单个实验

如果需要更细粒度的控制：

```bash
# 运行单个城市、单个种子的实验
python run_downstream_only.py \
    --task segment \
    --model HRNR_Hyperbolic \
    --dataset bj \
    --seed 31 \
    --output_dim 128 \
    --device gpu \
    --exp_id 1
```

## 结果分析

### 自动分析（Python脚本生成的结果）

如果使用`run_hrnr_experiments.py`，结果会自动保存为：

```
experiment_results/
├── raw_results_20260203_120000.json          # 原始结果
├── aggregated_results_20260203_120000.json   # 聚合结果（均值±标准差）
├── results_table_20260203_120000.csv         # CSV表格
└── report_20260203_120000.txt                # 文本报告
```

### 手动分析（Bash脚本或手动运行的结果）

如果使用bash脚本或手动运行，使用分析脚本提取结果：

```bash
# 分析指定目录的结果
python analyze_results.py ./experiment_results/20260203_120000

# 生成CSV格式摘要
python analyze_results.py ./experiment_results/20260203_120000 --format csv --output summary.csv

# 生成Markdown格式报告
python analyze_results.py ./experiment_results/20260203_120000 --format markdown --output report.md

# 生成LaTeX表格
python analyze_results.py ./experiment_results/20260203_120000 --format latex --output table.tex

# 生成JSON格式（包含所有详细信息）
python analyze_results.py ./experiment_results/20260203_120000 --format json --output results.json

# 详细模式（显示每个文件的处理过程）
python analyze_results.py ./experiment_results/20260203_120000 --verbose
```

## 结果格式

### CSV表格格式

```csv
Dataset,Model,Task,Num_Runs,speed_mae_mean,speed_mae_std,speed_rmse_mean,speed_rmse_std,...
bj,HRNR_Hyperbolic,segment,5,1.2345,0.0123,1.5678,0.0234,...
cd,HRNR_Hyperbolic,segment,5,1.3456,0.0145,1.6789,0.0256,...
...
```

### 文本报告格式

```
================================================================================
Dataset: BJ
================================================================================
Number of successful runs: 5/5

Speed Inference:
  speed_mae          : 1.2345 ± 0.0123
  speed_rmse         : 1.5678 ± 0.0234
  speed_mape         : 8.9012 ± 0.1234

Travel Time Estimation:
  travel_time_mae    : 2.3456 ± 0.0345
  travel_time_rmse   : 3.4567 ± 0.0456
  travel_time_mape   : 12.3456 ± 0.2345

Similarity Search:
  similarity_p10     : 0.8901 ± 0.0123
  similarity_r10     : 0.7890 ± 0.0234
```

### Markdown表格格式

适合直接复制到论文或文档中：

```markdown
## Dataset: BJ

### Speed Inference
| Metric | Mean | Std | Min | Max | Count |
|--------|------|-----|-----|-----|-------|
| speed_mae | 1.2345 | 0.0123 | 1.2100 | 1.2600 | 5 |
| speed_rmse | 1.5678 | 0.0234 | 1.5300 | 1.6100 | 5 |
```

### LaTeX表格格式

可直接插入LaTeX论文：

```latex
\begin{table}[htbp]
\centering
\caption{HRNR\_Hyperbolic Experimental Results}
\label{tab:hrnr_results}
\begin{tabular}{lcccc}
\hline
Dataset & Metric & Mean & Std & Count \\
\hline
bj & speed_mae & 1.2345 & 0.0123 & 5 \\
 & speed_rmse & 1.5678 & 0.0234 & 5 \\
\hline
\end{tabular}
\end{table}
```

## 实验流程示例

完整的实验流程：

```bash
# 1. 确认环境
python --version  # Python 3.7+
pip list | grep torch  # PyTorch已安装

# 2. 检查embeddings
ls -lh ./veccity/cache/1/evaluate_cache/road_embedding_HRNR_Hyperbolic_*_128.npy

# 3. 运行实验（Python脚本，推荐）
python run_hrnr_experiments.py --task segment --device gpu

# 或使用bash脚本
chmod +x run_hrnr_experiments.sh
./run_hrnr_experiments.sh gpu 1

# 4. 等待实验完成（可能需要数小时）
# 每个数据集×每个种子 = 5×5 = 25个实验
# 预计每个实验10-30分钟，总计约5-12小时

# 5. 查看结果
cat ./experiment_results/*/report_*.txt

# 6. 生成不同格式的报告
python analyze_results.py ./experiment_results/20260203_120000 --format csv
python analyze_results.py ./experiment_results/20260203_120000 --format markdown
python analyze_results.py ./experiment_results/20260203_120000 --format latex
```

## 故障排除

### 问题1: CUDA OOM错误

如果遇到GPU内存不足错误：

```bash
# 方法1: 使用CPU
python run_hrnr_experiments.py --device cpu

# 方法2: 减少数据集数量
python run_hrnr_experiments.py --datasets bj cd --device gpu

# 方法3: 设置CUDA内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_hrnr_experiments.py --device gpu
```

### 问题2: Embeddings文件未找到

```bash
# 检查文件是否存在
ls -lh ./veccity/cache/*/evaluate_cache/

# 如果不存在，需要先训练模型
python run_model.py --task segment --model HRNR_Hyperbolic --dataset bj --device gpu
```

### 问题3: 权限问题

```bash
# 添加执行权限
chmod +x run_hrnr_experiments.sh

# 检查目录权限
ls -la ./veccity/cache/
```

### 问题4: 某些实验失败

```bash
# 查看失败的实验日志
grep -r "FAILED" ./experiment_results/*/experiment_log.txt

# 手动重新运行失败的实验
python run_downstream_only.py --task segment --model HRNR_Hyperbolic --dataset bj --seed 31 --device gpu
```

## 高级用法

### 并行运行实验（使用GNU Parallel）

如果有多个GPU，可以并行运行：

```bash
# 安装GNU Parallel
sudo apt-get install parallel  # Ubuntu/Debian
brew install parallel           # macOS

# 并行运行不同数据集
parallel -j 2 "python run_downstream_only.py --task segment --model HRNR_Hyperbolic --dataset {} --seed 31 --device gpu --exp_id 1" ::: bj cd prt xa df
```

### 自定义评估任务

```bash
# 只运行特定的下游任务
python run_downstream_only.py \
    --task segment \
    --model HRNR_Hyperbolic \
    --dataset bj \
    --seed 31 \
    --device gpu \
    --evaluate_task speed_inference  # 只运行速度推断
```

### 批量处理结果

```bash
# 分析所有实验结果目录
for dir in ./experiment_results/*/; do
    echo "Analyzing $dir"
    python analyze_results.py "$dir" --format csv --output "${dir}/summary.csv"
done

# 合并所有CSV结果
head -1 ./experiment_results/*/summary.csv | head -1 > all_results.csv
tail -n +2 -q ./experiment_results/*/summary.csv >> all_results.csv
```

## 注意事项

1. **内存管理**: 确保有足够的GPU内存（建议16GB+）
2. **时间预算**: 完整实验可能需要5-12小时
3. **结果备份**: 定期备份`./veccity/cache/`和`./experiment_results/`目录
4. **日志查看**: 实验过程中可以查看日志文件了解进度
5. **中断恢复**: 如果实验中断，可以手动重新运行失败的部分

## 论文中的表格生成

使用分析脚本生成符合论文格式的表格：

```bash
# 生成LaTeX表格
python analyze_results.py ./experiment_results/20260203_120000 --format latex --output paper_table.tex

# 生成Markdown表格（用于初稿）
python analyze_results.py ./experiment_results/20260203_120000 --format markdown --output paper_table.md
```

然后在论文中：

```latex
% 在论文LaTeX文件中
\input{paper_table.tex}
```

## 参考

- HRNR论文: [链接]
- VecCity框架: [链接]
- 实验设置说明: 见论文第X节

## 联系方式

如有问题，请联系：[你的邮箱]
