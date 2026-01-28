# 训练和评估分离指南

本指南说明如何将**预训练（Pretraining）**和**下游任务评估（Downstream Evaluation）**分开运行，提高开发效率。

---

## 🎯 使用场景

### 场景1：完整流程（训练+评估）
适用于首次运行，需要完整的训练和评估流程。

```bash
# 使用原始run.py
python run.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge
```

### 场景2：仅训练模型（生成embeddings）
适用于：
- 只想训练模型，不运行耗时的下游任务
- 快速测试模型训练是否正常
- 批量训练多个模型，稍后统一评估

```bash
# 使用新的run_training_only.py
python run_training_only.py \
    --task segment \
    --model HRNR_Hyperbolic \
    --dataset bj_roadmap_edge \
    --device cuda \
    --saved_model
```

### 场景3：仅运行下游任务（使用已有embeddings）
适用于：
- Embeddings已经生成，只需要评估下游任务
- 调试下游任务代码
- 测试不同的下游任务配置
- 对比不同模型的下游任务性能

```bash
# 使用新的run_downstream_only.py
python run_downstream_only.py \
    --task segment \
    --model HRNR_Hyperbolic \
    --dataset bj_roadmap_edge \
    --exp_id 12345 \
    --device cuda
```

---

## 📁 文件说明

### 1. `run_training_only.py` - 仅训练脚本

**功能**：
- 训练模型并生成embeddings
- 保存模型checkpoint（可选）
- **不运行**下游任务评估

**主要参数**：
```bash
python run_training_only.py \
    --task segment \              # 任务类型：segment/region/poi
    --model HRNR_Hyperbolic \     # 模型名称
    --dataset bj_roadmap_edge \   # 数据集名称
    --device cuda \               # 设备：cpu/cuda
    --seed 31 \                   # 随机种子
    --exp_id 12345 \              # 实验ID（可选，默认自动生成）
    --saved_model                 # 保存模型checkpoint
```

**输出文件**：
```
./veccity/cache/{exp_id}/
├── model_cache/
│   └── HRNR_Hyperbolic_bj_roadmap_edge.m     # 模型checkpoint
└── evaluate_cache/
    └── road_embedding_HRNR_Hyperbolic_bj_roadmap_edge_128.npy  # 预训练embeddings
```

---

### 2. `run_downstream_only.py` - 仅评估脚本

**功能**：
- 加载已有的预训练embeddings
- 运行下游任务评估（Speed Inference, Travel Time Estimation等）
- **不重新训练**模型

**主要参数**：
```bash
python run_downstream_only.py \
    --task segment \              # 任务类型：segment/region/poi
    --model HRNR_Hyperbolic \     # 模型名称
    --dataset bj_roadmap_edge \   # 数据集名称
    --exp_id 12345 \              # 实验ID（可选，自动查找）
    --output_dim 128 \            # Embedding维度
    --device cuda \               # 设备：cpu/cuda
    --evaluate_task speed_inference travel_time_estimation  # 下游任务列表
```

**自动查找embeddings**：
如果不指定 `--exp_id`，脚本会自动搜索匹配的embedding文件：
```bash
python run_downstream_only.py \
    --task segment \
    --model HRNR_Hyperbolic \
    --dataset bj_roadmap_edge
```

**输出文件**：
```
./raw_data/new/evaluate_cache/
└── {exp_id}_evaluate_{exp_id}_{model}_{dataset}.csv  # 评估结果
```

---

## ⚡ GPU 加速使用

### 设备参数说明

两个脚本都支持多种设备参数格式：

```bash
# 方式1: 使用 'gpu' (自动使用 cuda:0)
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset xa --device gpu

# 方式2: 使用 'cuda' (自动使用 cuda:0)
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset xa --device cuda

# 方式3: 指定GPU编号 'cuda:0', 'cuda:1', etc.
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset xa --device cuda:0
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset xa --device cuda:1

# 方式4: 使用CPU
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset xa --device cpu
```

### GPU加速效果

| 操作 | CPU耗时 | GPU耗时 | 加速比 |
|------|---------|---------|--------|
| 模型训练 (1000 steps) | ~2-4小时 | **~10-20分钟** | **6-12x** |
| TTE下游任务 (100 epochs) | ~5分钟 | **~30秒** | **10x** |
| STS下游任务 (50 epochs) | ~10分钟 | **~1分钟** | **10x** |

### 自动GPU检测

脚本会自动：
1. 检测 CUDA 是否可用
2. 设置正确的 GPU 设备
3. 将模型和数据移到 GPU
4. 如果 GPU 不可用，自动回退到 CPU

**示例输出**：
```
Device: cuda
CUDA Available: True
Current Device: cuda:0
```

### 多GPU支持

如果有多个GPU，可以指定使用哪个：

```bash
# 使用第一个GPU (cuda:0)
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset xa --device cuda:0

# 使用第二个GPU (cuda:1)
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset xa --device cuda:1

# 查看可用GPU
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

---

## 🚀 典型工作流

### 工作流1：批量训练多个模型（GPU加速）

```bash
# 使用GPU加速训练多个模型
python run_training_only.py --task segment --model HRNR --dataset xa --device cuda --exp_id 10001
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset xa --device cuda --exp_id 10002
python run_training_only.py --task segment --model HyperRoad --dataset xa --device cuda --exp_id 10003

# GPU加速下游任务评估
python run_downstream_only.py --task segment --model HRNR --dataset xa --device cuda --exp_id 10001
python run_downstream_only.py --task segment --model HRNR_Hyperbolic --dataset xa --device cuda --exp_id 10002
python run_downstream_only.py --task segment --model HyperRoad --dataset xa --device cuda --exp_id 10003
```

### 工作流1（原版）：批量训练多个模型

```bash
# 1. 训练多个模型（不运行下游任务）
python run_training_only.py --task segment --model HRNR --dataset bj_roadmap_edge --exp_id 10001
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge --exp_id 10002
python run_training_only.py --task segment --model HyperRoad --dataset bj_roadmap_edge --exp_id 10003

# 2. 统一运行下游任务评估（快速）
python run_downstream_only.py --task segment --model HRNR --dataset bj_roadmap_edge --exp_id 10001
python run_downstream_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge --exp_id 10002
python run_downstream_only.py --task segment --model HyperRoad --dataset bj_roadmap_edge --exp_id 10003
```

### 工作流2：调试下游任务

```bash
# 1. 训练一次，生成embeddings
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge --exp_id 10001

# 2. 多次运行下游任务（修改代码后测试）
python run_downstream_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge --exp_id 10001

# 修改下游任务代码...

python run_downstream_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge --exp_id 10001

# 再次修改...

python run_downstream_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge --exp_id 10001
```

### 工作流3：测试不同下游任务配置

```bash
# 使用已有embeddings，测试不同任务组合
python run_downstream_only.py \
    --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge \
    --exp_id 10001 \
    --evaluate_task speed_inference

python run_downstream_only.py \
    --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge \
    --exp_id 10001 \
    --evaluate_task travel_time_estimation

python run_downstream_only.py \
    --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge \
    --exp_id 10001 \
    --evaluate_task speed_inference travel_time_estimation
```

---

## ⚡ 性能对比

### 原始流程（训练+评估）
```
训练模型 (2-4小时) + 下游任务评估 (优化前：数小时 → 优化后：几分钟)
总时间：2-4小时 + 评估时间
```

### 分离流程（训练 | 评估）
```bash
# 一次训练
训练模型 (2-4小时)  → 保存embeddings

# 多次评估（无需重新训练）
评估下游任务 (几分钟) ✅
评估下游任务 (几分钟) ✅  ← 修改代码后重新评估
评估下游任务 (几分钟) ✅  ← 再次修改后重新评估
```

**优势**：
- 🚀 下游任务调试速度提升 **100-1000倍**
- 💾 避免重复训练，节省计算资源
- 🔧 方便快速迭代和测试

---

## 📊 Embeddings文件管理

### 查找已有embeddings

```bash
# 查看所有实验ID
ls ./veccity/cache/

# 查看特定实验的embeddings
ls ./veccity/cache/12345/evaluate_cache/

# 搜索特定模型的embeddings
find ./veccity/cache -name "road_embedding_HRNR_Hyperbolic_*.npy"
```

### Embeddings命名规则

```
road_embedding_{model}_{dataset}_{output_dim}.npy

示例：
road_embedding_HRNR_Hyperbolic_bj_roadmap_edge_128.npy
road_embedding_HRNR_bj_roadmap_edge_64.npy
```

---

## 🛠️ 故障排查

### 问题1: Embedding文件找不到

**错误信息**：
```
FileNotFoundError: No embedding file found for model=HRNR_Hyperbolic
```

**解决方法**：
```bash
# 检查embeddings是否存在
find ./veccity/cache -name "road_embedding_HRNR_Hyperbolic_*.npy"

# 如果不存在，先运行训练
python run_training_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge
```

### 问题2: 实验ID不匹配

**错误信息**：
```
Embedding file not found: ./veccity/cache/12345/evaluate_cache/...
```

**解决方法**：
```bash
# 不指定exp_id，让脚本自动查找
python run_downstream_only.py --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge

# 或者查看训练时输出的exp_id
grep "Experiment ID" ./veccity/log/*.log
```

### 问题3: 维度不匹配

**错误信息**：
```
ValueError: shapes not aligned: expected 64, got 128
```

**解决方法**：
```bash
# 检查embeddings的实际维度
python -c "import numpy as np; emb = np.load('path/to/embedding.npy'); print(emb.shape)"

# 使用正确的output_dim
python run_downstream_only.py \
    --task segment --model HRNR_Hyperbolic --dataset bj_roadmap_edge \
    --output_dim 64  # 修改为实际维度
```

---

## 📝 配置文件支持

两个脚本都支持使用配置文件覆盖默认设置：

```bash
# 使用自定义配置文件
python run_training_only.py \
    --task segment \
    --model HRNR_Hyperbolic \
    --dataset bj_roadmap_edge \
    --config_file custom_config.json

python run_downstream_only.py \
    --task segment \
    --model HRNR_Hyperbolic \
    --dataset bj_roadmap_edge \
    --config_file custom_config.json
```

---

## 🔗 与EmbeddingWrapper的配合

这些脚本配合新的 `EmbeddingWrapper` 类使用，确保下游任务：
- ✅ 使用预计算的embeddings（查表操作）
- ❌ 而不是重新运行模型前向传播

**性能提升**：
- TTE任务：从数小时 → 几秒钟
- STS任务：从数小时 → 几秒钟
- Speed Inference：几乎即时

---

## 💡 最佳实践

1. **开发阶段**：使用 `run_training_only.py` 训练模型，然后用 `run_downstream_only.py` 快速迭代下游任务

2. **批量实验**：先批量运行训练，再统一评估，提高资源利用率

3. **保存checkpoints**：训练时加 `--saved_model` 参数，方便后续加载

4. **使用GPU**：训练和评估都可以用 `--device cuda` 加速

5. **记录exp_id**：训练完成后记下exp_id，方便后续评估

---

## 📧 支持

如有问题，请查看日志文件：
```bash
./veccity/log/
```

或查看缓存目录结构：
```bash
tree ./veccity/cache/
```
