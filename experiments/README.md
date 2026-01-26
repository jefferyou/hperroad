# HRNR_Hyperbolic 实验脚本

完整的benchmark实验框架，支持多城市、多GPU加速训练。

## 快速开始

### 1. 运行完整Benchmark（推荐）

```bash
cd ~/Mingjie/hperroad/experiments
bash run_full_benchmark.sh
```

这将自动完成：
- ✓ 应用所有性能优化（预加载嵌入、DataLoader、多GPU）
- ✓ 在4个城市上运行完整实验（Xi'an, Beijing, Chengdu, San Francisco）
- ✓ 每个城市：预训练（100 epochs）+ 3个下游任务（TSI, TTE, STS）
- ✓ 使用5个GPU（3,4,5,6,7）进行DataParallel加速

**预期时间：** 16-20小时（4个城市）

### 2. 单独运行某个阶段

#### 仅预训练

```bash
python run_training_only.py \
    --dataset xa \
    --seed 0 \
    --max_epoch 100 \
    --gpu True \
    --gpu_id 0 \
    --train_gpu_ids 0 1 2 3 4
```

#### 仅评估

```bash
# 获取实验ID
ls -t ../VecCity-main/veccity/cache/ | grep "hrnr_hyp_xa"

# 运行评估
python run_evaluation_only.py \
    --exp_id hrnr_hyp_xa_s0_20260125_145753 \
    --task all \
    --task_epoch 100 \
    --gpu_id 0 \
    --eval_gpu_ids 0 1 2 3 4
```

## 文件说明

### 核心脚本

| 文件 | 说明 |
|------|------|
| `run_full_benchmark.sh` | **完整benchmark脚本**（推荐使用） |
| `run_training_only.py` | 独立的预训练脚本 |
| `run_evaluation_only.py` | 独立的评估脚本 |
| `apply_all_optimizations.py` | **统一优化脚本**（自动应用所有优化） |

### 优化内容

`apply_all_optimizations.py` 会自动应用以下优化：

1. **预加载嵌入优化** (100-1000x加速)
   - 修改 `hhgcl_evaluator.py`
   - 嵌入一次性加载到GPU，避免重复运行GNN

2. **DataLoader优化** (2-3x加速)
   - 修改 `travel_time_estimation.py` 和 `similarity_search_model.py`
   - batch_size: 128 → 512
   - num_workers: 4 → 8
   - 启用 pin_memory 和 persistent_workers

3. **多GPU支持** (3-4x加速)
   - 修改 `twostep_executor.py`
   - 使用PyTorch DataParallel分发训练到多个GPU

4. **DataParallel兼容性**
   - 自动处理DataParallel包装的模型
   - 添加 `_get_model()` 辅助方法

## 性能基准

### 单个城市（Xi'an为例）

| 阶段 | 单GPU时间 | 5-GPU时间 | 加速比 |
|------|-----------|-----------|--------|
| 预训练 (100 epochs) | ~12小时 | **3-3.5小时** | **3.4-4x** |
| TSI | <1分钟 | <1分钟 | 1x |
| TTE (100 epochs) | ~40小时 | **30-60分钟** | **40-80x** |
| STS (50 epochs) | ~5.8小时 | **10-20分钟** | **17-35x** |
| **总计** | ~58小时 | **~4-5小时** | **11-15x** |

### 完整Benchmark（4个城市）

- **原始时间：** ~232小时（9.7天）
- **优化后：** **16-20小时**
- **加速比：** **11-15x**

## GPU配置

### 查看GPU状态

```bash
watch -n 1 nvidia-smi
```

### 预期GPU使用

- **GPU利用率：** 70-90%
- **GPU功耗：** 200-240W（Tesla V100S-32GB）
- **显存使用：** 15-20GB/GPU

### 自定义GPU配置

编辑 `run_full_benchmark.sh`:

```bash
TRAIN_GPUS="3,4,5,6,7"  # 修改为你想用的GPU
EVAL_GPUS="3,4,5,6,7"
```

## 监控和日志

### 查看实时日志

```bash
# 查看最新的benchmark结果
ls -ltr results/

# 查看特定城市的TTE训练日志
tail -f results/benchmark_20260125_145751/xian_tte.log
```

### 结果文件位置

```
experiments/
├── results/
│   └── benchmark_TIMESTAMP/
│       ├── xian_training.log    # Xi'an预训练日志
│       ├── xian_tsi.log         # Xi'an TSI任务日志
│       ├── xian_tte.log         # Xi'an TTE任务日志
│       ├── xian_sts.log         # Xi'an STS任务日志
│       ├── beijing_*.log
│       ├── chengdu_*.log
│       └── sanfrancisco_*.log
```

### CSV结果文件

评估完成后，结果会保存在：

```
../VecCity-main/veccity/cache/EXPERIMENT_ID/evaluate_cache/
├── SpeedInferenceModel_tsi_xa.csv
├── TravelTimeEstimationModel_tte_xa.csv
└── SimilaritySearchModel_sts_xa.csv
```

## 常见问题

### 1. GPU内存不足

如果遇到 `CUDA out of memory` 错误，编辑 `apply_all_optimizations.py`:

```python
# 第113行左右，减小batch_size
content = content.replace(
    'batch_size=128',
    'batch_size=256'  # 从512改为256
)
```

然后重新应用优化：

```bash
python apply_all_optimizations.py
```

### 2. 恢复原始文件

所有修改的文件都有 `.backup_clean` 备份：

```bash
# 恢复单个文件
cp ../VecCity-main/veccity/executor/twostep_executor.py.backup_clean \
   ../VecCity-main/veccity/executor/twostep_executor.py

# 恢复所有文件
find ../VecCity-main -name "*.backup_clean" | while read backup; do
    original="${backup%.backup_clean}"
    cp "$backup" "$original"
done
```

### 3. 清除Python缓存

如果代码修改后没有生效：

```bash
find ../VecCity-main -name "*.pyc" -delete
find ../VecCity-main -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

### 4. AttributeError: 'TwoStepExecutor' object has no attribute 'cache_dir'

这通常是因为 `twostep_executor.py` 结构损坏。修复方法：

```bash
# 从备份恢复
cp ../VecCity-main/veccity/executor/twostep_executor.py.backup_clean \
   ../VecCity-main/veccity/executor/twostep_executor.py

# 重新应用优化
python apply_all_optimizations.py

# 清除缓存
find ../VecCity-main -name "*.pyc" -delete
find ../VecCity-main -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

### 5. 只运行某些城市

编辑 `run_full_benchmark.sh`:

```bash
# 第14行
DATASET_MAP=("xian:xa" "beijing:bj")  # 只运行Xi'an和Beijing
```

## 技术细节

### DataParallel工作原理

```
单GPU模式:
  model.run() → 正常调用

多GPU模式 (DataParallel):
  torch.nn.DataParallel(model) →
    DataParallel(
      (module): HRNR_Hyperbolic(...)
    )
  model.forward() → 自动分发（内置方法）
  model.run() → ❌ 错误（自定义方法不可见）
  model.module.run() → ✅ 正确（通过module访问）

解决方案:
  def _get_model(self):
      if isinstance(self.model, torch.nn.DataParallel):
          return self.model.module
      return self.model
```

### CUDA_VISIBLE_DEVICES

```bash
# 物理GPU: 0, 1, 2, 3, 4, 5, 6, 7
# 设置可见GPU为3,4,5,6,7
CUDA_VISIBLE_DEVICES=3,4,5,6,7

# 映射后的逻辑GPU: 0, 1, 2, 3, 4
# device_ids=[0,1,2,3,4]  ✓ 正确
# device_ids=[3,4,5,6,7]  ✗ 错误（超出逻辑范围）
```

## 实验配置（匹配HRNR论文）

| 参数 | 值 |
|------|-----|
| 预训练轮数 | 100 |
| TTE任务轮数 | 100 |
| STS任务轮数 | 50 |
| TSI任务轮数 | 10（Ridge回归，快速）|
| 城市 | Xi'an, Beijing, Chengdu, San Francisco |
| GPU数量 | 5 (Tesla V100S-32GB) |

## 联系方式

遇到问题？检查：
1. 日志文件：`results/benchmark_*/`
2. Python缓存：`find ../VecCity-main -name "*.pyc"`
3. GPU状态：`nvidia-smi`
4. 备份文件：`find ../VecCity-main -name "*.backup_clean"`
