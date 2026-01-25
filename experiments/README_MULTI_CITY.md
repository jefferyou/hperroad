# 多城市多GPU加速实验指南

## 📋 概述

自动化运行HRNR_Hyperbolic在多个城市数据集上的完整实验流程：
- **预训练**：学习双曲空间road embeddings
- **下游任务**：TSI (速度推断) + TTE (时间估计) + STS (相似度搜索)
- **多GPU加速**：下游任务使用DataParallel加速
- **性能优化**：预加载embeddings避免重复GNN计算

## 🗺️ 支持的城市

根据论文Table 4，支持4个城市：

| 城市 | 数据集代码 | 说明 |
|------|-----------|------|
| Xi'an | xa | 西安 (已测试) |
| Beijing | bj | 北京 |
| Chengdu | cd | 成都 |
| San Francisco | sf | 旧金山 |

## 🚀 快速开始

### 方法1：使用默认配置运行所有城市

```bash
cd ~/Mingjie/hperroad/experiments

# 1. 应用性能优化补丁
python apply_downstream_optimizations.py

# 2. 运行所有4个城市的完整实验
python run_multi_city_experiments.py
```

### 方法2：自定义城市和GPU配置

```bash
# 只运行Xi'an和Beijing
python run_multi_city_experiments.py --cities xian beijing

# 指定训练和评估使用的GPU
python run_multi_city_experiments.py \
    --train_gpu 3 \
    --eval_gpus 3 4 5 6 7

# 减少epoch数（快速测试）
python run_multi_city_experiments.py \
    --max_epoch 10 \
    --task_epoch 10 \
    --cities xian
```

### 方法3：单个城市独立运行

```bash
# Step 1: 预训练
python run_training_only.py \
    --dataset xa \
    --seed 0 \
    --max_epoch 100 \
    --gpu True \
    --gpu_id 3

# Step 2: 下游任务评估（使用多GPU）
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python run_evaluation_only.py \
    --exp_id <your_exp_id> \
    --task all \
    --task_epoch 100 \
    --gpu_id 0
```

## ⚙️ 配置文件

编辑 `experiment_config.json` 自定义实验配置：

```json
{
  "training": {
    "max_epoch": 100,      // 预训练轮数（HRNR默认100）
    "seed": 0
  },
  "downstream_tasks": {
    "tte": {
      "epochs": 100        // HRNR默认TTE=100轮
    },
    "sts": {
      "epochs": 50         // HRNR默认STS=50轮
    }
  },
  "gpu_config": {
    "train_gpu": 3,                    // 预训练使用GPU 3
    "eval_gpus": [3, 4, 5, 6, 7]       // 下游任务使用5个GPU并行
  }
}
```

## 🎯 性能优化详解

### 1. 预加载Embeddings（核心优化）

**问题：** 原始代码每个batch都重新运行GNN
```python
# 原始代码 - 慢！
for batch in dataloader:  # 1772 batches
    emb = model.graph_enc(...)  # 每次都计算全图！→ 运行GNN 1772次
    output = downstream_model(emb[batch])
```

**优化：** 预加载到GPU，直接索引
```python
# 优化后 - 快！
emb = np.load('embeddings.npy')
emb_gpu = torch.from_numpy(emb).cuda()  # 一次性加载到GPU

for batch in dataloader:
    output = downstream_model(emb_gpu[batch])  # 直接GPU索引
```

**效果：** 100-1000倍加速！

### 2. 多GPU并行（DataParallel）

```python
model = nn.DataParallel(model, device_ids=[3,4,5,6,7])
```

- 自动将batch分配到5个GPU
- 5倍吞吐量提升

### 3. DataLoader优化

```python
DataLoader(
    dataset,
    batch_size=512,           # ↑ 充分利用32GB显存
    num_workers=8,            # ↑ Linux多进程高效
    pin_memory=True,          # ↑ 加速CPU→GPU传输
    persistent_workers=True   # ↑ 复用worker进程
)
```

## 📊 预期性能

### 单个城市完整流程

| 阶段 | 原始时间 | 优化后时间 | 加速比 |
|------|----------|------------|--------|
| 预训练 (100 epochs) | ~12小时 | ~12小时 | 1x |
| TSI | <1分钟 | <1分钟 | 1x |
| TTE (100 epochs) | ~40小时 | **30-60分钟** | **40-80x** |
| STS (50 epochs) | ~5.8小时 | **10-20分钟** | **17-35x** |
| **总计** | **~58小时** | **~13-14小时** | **4-4.5x** |

### 4个城市总耗时

- **原始**: ~232小时 (9.7天)
- **优化后**: ~52-56小时 (2.2-2.3天)
- **加速比**: 4-4.5倍

### GPU利用率

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| GPU利用率 | 1% | 70-90% |
| GPU功耗 | 38-57W | 200-240W |
| 显存使用 | 1.8-7GB | 15-20GB/GPU |

## 📁 输出结果

实验完成后生成：

```
experiments/results/
├── multi_city_summary_20260125_XXXXXX.json   # 实验总结
└── <exp_id>/
    ├── model_cache/
    │   └── HRNR_Hyperbolic_<dataset>.m       # 训练的模型
    ├── evaluate_cache/
    │   ├── road_embedding_*.npy              # Road embeddings
    │   └── *_evaluate_*.csv                  # 下游任务结果
    └── logs/
        └── *.log                             # 训练日志
```

### 结果JSON示例

```json
{
  "total_duration_hours": 54.3,
  "cities_completed": 4,
  "results": {
    "xian": {
      "status": "SUCCESS",
      "exp_id": "hrnr_hyp_xa_s0_20260125_103045",
      "duration_hours": 13.2
    },
    "beijing": {
      "status": "SUCCESS",
      "exp_id": "hrnr_hyp_bj_s0_20260125_233015",
      "duration_hours": 14.1
    }
    // ...
  }
}
```

## 🔧 故障排查

### 问题1：CUDA out of memory

**解决：** 减小batch size
```bash
# 编辑 apply_downstream_optimizations.py
# 将 batch_size=512 改为 batch_size=256 或 128
```

### 问题2：找不到数据集

**检查：** 数据集是否存在
```bash
ls ~/Mingjie/hperroad/VecCity-main/raw_data/
# 应该看到：bj/, cd/, xa/, sf/
```

### 问题3：GPU被其他进程占用

**解决：** 指定空闲GPU
```bash
# 先查看GPU状态
nvidia-smi

# 使用空闲GPU
python run_multi_city_experiments.py \
    --train_gpu 4 \
    --eval_gpus 5 6 7
```

### 问题4：优化补丁失败

**手动应用：**
```bash
# 恢复原始文件
cp file.py.backup file.py

# 重新运行补丁
python apply_downstream_optimizations.py
```

## 💡 最佳实践

1. **先测试单个城市**
   ```bash
   python run_multi_city_experiments.py --cities xian --max_epoch 10 --task_epoch 10
   ```

2. **使用screen/tmux避免SSH断线**
   ```bash
   screen -S hrnr_experiments
   python run_multi_city_experiments.py
   # Ctrl+A, D 离开screen
   # screen -r hrnr_experiments 重新连接
   ```

3. **监控GPU使用**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **保存实验日志**
   ```bash
   python run_multi_city_experiments.py 2>&1 | tee experiment.log
   ```

## 📚 参考

- 原始HRNR论文：Table 4 (性能对比)
- VecCity框架文档
- PyTorch DataParallel文档

## 🆘 获取帮助

```bash
python run_multi_city_experiments.py --help
```
