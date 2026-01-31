# GPU完全配置指南

## 🎯 目标

确保HRNR和HRNR_Hyperbolic的**整个流程**都使用GPU，包括：
1. 模型训练阶段
2. 下游任务评估（STS, TTE, TSI）
3. 所有tensor操作

---

## ✅ 已完成的配置修改

### 1. 配置文件更新

**修改内容**：
```json
// 之前（错误）
{
  "gpu": false,
  "device": "cpu"
}

// 现在（正确）
{
  "gpu": true,
  "gpu_id": 0,
  "device": "cuda"
}
```

**修改的文件**：
- `/VecCity-main/veccity/config/model/segment/HRNR.json`
- `/VecCity-main/veccity/config/model/segment/HRNR_Hyperbolic.json`

### 2. 实验脚本GPU参数传递

所有实验脚本已配置为传递GPU参数：

```python
other_args = {
    'gpu': True,      # 启用GPU
    'gpu_id': 0,      # GPU设备ID
    'device': 'cuda', # PyTorch设备
    ...
}
```

---

## 🔍 验证GPU配置

### 方式1：运行验证脚本

```bash
cd /home/user/hperroad/experiments
python verify_gpu_config.py
```

这会检查：
- ✅ PyTorch GPU可用性
- ✅ 配置文件GPU设置
- ✅ 实验脚本参数传递
- ✅ 生成配置总结文档

### 方式2：手动检查

#### 检查PyTorch GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

预期输出：
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA GeForce RTX 5070 Ti
```

#### 检查配置文件

```bash
# HRNR_Hyperbolic配置
grep -E "gpu|device" /home/user/hperroad/VecCity-main/veccity/config/model/segment/HRNR_Hyperbolic.json

# 预期输出：
# "gpu": true,
# "gpu_id": 0,
# "device": "cuda",
```

---

## 🚀 运行实验并监控GPU

### 1. 启动实验

```bash
cd /home/user/hperroad/experiments

# 方式1：使用启动脚本（推荐）
./run_ablation.sh --gpu-id 0

# 方式2：直接Python
python run_complete_ablation.py --dataset xa --seed 0 --gpu true --gpu_id 0
```

### 2. 实时监控GPU使用

**在另一个终端窗口**运行：

```bash
# 持续监控（推荐）
watch -n 1 nvidia-smi

# 或单次查看
nvidia-smi
```

### 3. 验证GPU使用情况

**正常使用GPU时，应该看到：**

```
+-----------------------------------------------------------------------------------------+
| GPU  Name                  ...  | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |    Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 Ti  |                  N/A |
| 50%   65C    P0            250W /  300W |  8000MiB / 16303MiB |    95%      Default |
```

**关键指标：**
- **Memory-Usage**: 应该增加（例如从1679MiB到8000MiB+）
- **GPU-Util**: 应该高（例如80-100%）
- **Power**: 应该接近功率上限（例如250W/300W）
- **Temp**: 温度上升（例如60-80°C）

**如果GPU未使用，会看到：**
```
| Memory-Usage: ~1679MiB (接近初始值)
| GPU-Util: 0-10% (几乎没有利用率)
| Power: ~30W (接近空闲功率)
```

---

## 🔧 确保全流程GPU使用的关键点

### 1. 模型训练阶段

**VecCity会根据配置自动设置设备：**

```python
# 在HRNR_Hyperbolic.py中
self.device = config.get("device", torch.device("cpu"))

# 所有tensor操作使用.to(self.device)
self.linear = torch.nn.Linear(...).to(self.device)
```

✅ **已配置**: `config["device"] = "cuda"`

### 2. 下游任务评估

下游任务（STS, TTE, TSI）会继承配置：

```python
# 在run_model中，other_args会合并到config
config.update(other_args)

# 因此gpu、gpu_id、device参数会传递到评估任务
```

✅ **已配置**: 实验脚本传递 `gpu=True, gpu_id=0`

### 3. 数据加载

确保数据也在GPU上：

```python
# 在训练循环中
train_set = train_set.clone().detach().to(self.device)
train_label = train_label.clone().detach().to(self.device)
```

✅ **已实现**: HRNR_Hyperbolic.py line 248-249

---

## 📊 GPU使用情况对比

### CPU模式（之前）
```
训练时间：~2-3小时/epoch
GPU利用率：0%
GPU内存：~1679MiB（空闲）
```

### GPU模式（现在）
```
训练时间：~5-10分钟/epoch（估计）
GPU利用率：80-100%
GPU内存：8000-12000MiB（取决于batch size）
```

**预期加速比**: 10-30x（取决于模型和batch size）

---

## 🐛 故障排除

### Q1: GPU利用率为0%

**原因**：配置未生效

**解决**：
```bash
# 1. 验证配置文件
python verify_gpu_config.py

# 2. 显式指定GPU
./run_ablation.sh --gpu-id 0

# 3. 检查日志中的device
# 应该看到 "device: cuda"
```

### Q2: CUDA out of memory

**原因**：batch size过大或GPU内存不足

**解决**：
```bash
# 减小batch size
# 编辑配置文件：HRNR_Hyperbolic.json
# "batch_size": 32  # 从64减小到32

# 或使用梯度累积
```

### Q3: GPU温度过高

**原因**：长时间满负载

**解决**：
```bash
# 1. 检查GPU风扇
nvidia-smi -q -d TEMPERATURE

# 2. 降低GPU频率（如果需要）
# 3. 确保机箱通风良好
```

### Q4: 多GPU选择

**如果有多个GPU**：

```bash
# 使用GPU 1而不是GPU 0
./run_ablation.sh --gpu-id 1

# 或在Python中
python run_complete_ablation.py --gpu_id 1
```

---

## 📈 性能优化建议

### 1. Batch Size调整

```json
// 根据GPU内存调整
{
  "batch_size": 64   // RTX 5070 Ti (16GB) 应该可以
  // 如果OOM，降到32或16
}
```

### 2. 混合精度训练（可选）

如果需要进一步加速，可以启用混合精度：

```python
# 在训练循环中
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    pred = self.encode(train_set)
    loss = criterion(pred, train_label)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 数据预加载

确保数据加载不成为瓶颈：

```python
# 在DataLoader中
num_workers=4,  # 多线程加载
pin_memory=True  # 加速CPU到GPU传输
```

---

## ✅ 验证清单

运行实验前，确认：

- [ ] `nvidia-smi` 显示GPU可用
- [ ] `python verify_gpu_config.py` 全部通过
- [ ] 配置文件中 `"gpu": true, "device": "cuda"`
- [ ] 启动脚本使用 `--gpu-id 0`
- [ ] 运行时监控 `watch -n 1 nvidia-smi`
- [ ] 看到GPU利用率80%+
- [ ] 看到GPU内存使用增加

---

## 📚 相关文档

- [QUICK_START_ABLATION.md](QUICK_START_ABLATION.md) - 快速开始
- [README_ABLATION.md](README_ABLATION.md) - 详细说明
- [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md) - 完整指南

---

## 🎯 快速开始（GPU模式）

```bash
# 1. 验证GPU配置
python verify_gpu_config.py

# 2. 运行实验
./run_ablation.sh --gpu-id 0

# 3. 监控GPU（另一个终端）
watch -n 1 nvidia-smi
```

---

**创建日期**: 2026-01-26
**GPU**: NVIDIA GeForce RTX 5070 Ti (16GB)
**CUDA版本**: 12.9
