# 快速使用指南 - 支持断点续传

## 第一步：拉取最新代码

```bash
cd ~/Mingjie/hperroad
git pull origin claude/hyperbolic-embeddings-veccity-9Rpvs
```

## 第二步：清除Python缓存（重要！）

```bash
find VecCity-main -name "*.pyc" -delete
find VecCity-main -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

## 第三步：运行Benchmark

```bash
cd experiments
bash run_full_benchmark.sh
```

就这么简单！

## ⭐ 新功能：自动断点续传

**脚本会自动保存进度！** 如果中途中断（停电、SSH断线、错误等），只需重新运行相同命令：

```bash
bash run_full_benchmark.sh
```

脚本会：
- ✅ 自动检测已完成的任务并跳过
- ✅ 从中断的地方继续运行
- ✅ 使用相同的结果目录
- ✅ 保留所有已完成的结果

### 示例：中断后恢复

```bash
# 首次运行
bash run_full_benchmark.sh
# 输出：Processing Xi'an... ✓ Training completed
#      Processing Beijing... ✗ TTE failed (interrupted)

# 稍后恢复（会自动跳过Xi'an）
bash run_full_benchmark.sh
# 输出：RESUMING FROM PREVIOUS RUN
#      [SKIP] Xi'an training already completed
#      Processing Beijing... [SKIP] Training already completed
#                          [SKIP] TSI already completed
#                          Starting TTE from beginning...
```

## 会发生什么？

1. **自动应用优化** (~5秒)
   - 预加载嵌入（100-1000x加速）
   - DataLoader优化（2-3x加速）
   - 多GPU支持（3-4x加速）

2. **处理Xi'an城市** (~4-5小时)
   - 预训练：100 epochs with 5 GPUs
   - TSI：<1分钟
   - TTE：100 epochs，~30-60分钟
   - STS：50 epochs，~10-20分钟

3. **处理Beijing城市** (~4-5小时)
4. **处理Chengdu城市** (~4-5小时)
5. **处理San Francisco城市** (~4-5小时)

**总时间：16-20小时**

## 监控运行

### 查看GPU使用

```bash
watch -n 1 nvidia-smi
```

应该看到5个GPU（3,4,5,6,7）全部在工作，利用率70-90%。

### 查看日志

```bash
# 查看最新的benchmark目录
ls -ltr results/

# 实时查看Xi'an的TTE训练
tail -f results/benchmark_*/xian_tte.log
```

## 进度管理

### 查看当前进度

```bash
# 查看状态文件
cat .benchmark_state

# 示例输出：
# RESULTS_DIR="results/benchmark_20260125_145751"
# xian_training=completed
# xian_exp_id="hrnr_hyp_xa_s0_20260125_145753"
# xian_tsi=completed
# xian_tte=completed
# xian_sts=completed
# beijing_training=completed
# beijing_exp_id="hrnr_hyp_bj_s0_20260125_152130"
```

### 重新开始（丢弃当前进度）

```bash
# 删除状态文件
rm .benchmark_state

# 重新运行
bash run_full_benchmark.sh
```

### 只运行特定城市

如果某个城市经常失败，可以编辑脚本只运行它：

```bash
# 编辑 run_full_benchmark.sh
nano run_full_benchmark.sh

# 第17行改为：
DATASET_MAP=("beijing:bj")  # 只运行Beijing

# 保存后运行
bash run_full_benchmark.sh
```

### 手动标记任务完成

如果你确定某个任务已经完成但脚本不识别：

```bash
# 在.benchmark_state中添加
echo "beijing_tte=completed" >> .benchmark_state
```

## 如果出错

### 1. cache_dir错误

```bash
# 恢复twostep_executor.py
cp ../VecCity-main/veccity/executor/twostep_executor.py.backup_clean \
   ../VecCity-main/veccity/executor/twostep_executor.py

# 重新优化
python apply_all_optimizations.py

# 清除缓存
find ../VecCity-main -name "*.pyc" -delete
find ../VecCity-main -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

### 2. GPU内存不足

```bash
# 编辑优化脚本，减小batch_size
nano apply_all_optimizations.py
# 将第113行的512改为256

# 重新优化
python apply_all_optimizations.py
```

### 3. 查看完整错误

```bash
# 查看最新的日志文件
cat results/benchmark_*/xian_training.log
```

## 结果在哪里？

### 日志文件

```
experiments/results/benchmark_TIMESTAMP/
├── xian_training.log
├── xian_tsi.log
├── xian_tte.log
├── xian_sts.log
└── ...
```

### CSV结果

```
VecCity-main/veccity/cache/EXPERIMENT_ID/evaluate_cache/
├── SpeedInferenceModel_tsi_xa.csv
├── TravelTimeEstimationModel_tte_xa.csv
└── SimilaritySearchModel_sts_xa.csv
```

## 常见问题

**Q: 脚本卡住不动？**
- 检查 `nvidia-smi`，确认GPU在工作
- 查看日志文件，看最后输出是什么

**Q: 想只跑一个城市？**
```bash
# 编辑 run_full_benchmark.sh
# 第14行改为：
DATASET_MAP=("xian:xa")
```

**Q: 想改变GPU数量？**
```bash
# 编辑 run_full_benchmark.sh
# 第19-20行改为：
TRAIN_GPUS="3,4,5"  # 只用3个GPU
EVAL_GPUS="3,4,5"
```

**Q: 需要帮助？**
- 查看完整文档：`cat README.md`
- 检查备份文件：`find ../VecCity-main -name "*.backup_clean"`
