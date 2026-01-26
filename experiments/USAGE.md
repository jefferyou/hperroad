# 快速使用指南

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
