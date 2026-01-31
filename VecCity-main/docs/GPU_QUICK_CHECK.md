# ⚡ GPU配置快速验证

## 📦 已完成的更改

✅ **配置文件已更新为GPU模式**

```json
// HRNR.json & HRNR_Hyperbolic.json
{
  "gpu": true,        // ✅ 已启用
  "gpu_id": 0,        // ✅ 已添加
  "device": "cuda"    // ✅ 已更改（原来是"cpu"）
}
```

---

## 🚀 立即验证

### 方式1：运行验证脚本（推荐）

```bash
cd /home/user/hperroad/experiments
python verify_gpu_config.py
```

这会自动检查：
- PyTorch GPU可用性
- 配置文件GPU设置
- 实验脚本参数传递
- 生成配置总结

### 方式2：快速手动检查

```bash
# 1. 检查GPU可用
nvidia-smi

# 2. 检查PyTorch CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# 3. 验证配置文件
grep -E "gpu|device" VecCity-main/veccity/config/model/segment/HRNR_Hyperbolic.json
```

**预期输出**：
```
"gpu": true,
"gpu_id": 0,
"device": "cuda",
```

---

## 🎯 运行实验

```bash
cd /home/user/hperroad/experiments

# 启动实验
./run_ablation.sh --gpu-id 0

# 在另一个终端监控GPU
watch -n 1 nvidia-smi
```

---

## 📊 GPU使用验证

**正常使用GPU时，`nvidia-smi`应显示：**

```
+-----------------------------------------------------------------------------+
| GPU  Name                    | Memory-Usage | GPU-Util  |
|=========================================+=============+===================|
|   0  NVIDIA GeForce RTX 5070 Ti  | 8000MB / 16GB |   95%     |  ← 高利用率
+-----------------------------------------------------------------------------+
```

**关键指标：**
- Memory-Usage: 应该从 ~1.6GB 增加到 8GB+
- GPU-Util: 应该在 80-100%
- Power: 应该接近 300W

**如果没有使用GPU：**
- Memory-Usage: 保持 ~1.6GB
- GPU-Util: 0-10%
- Power: ~30W

---

## 🔧 故障排除

### GPU利用率为0%？

```bash
# 1. 重新验证配置
python verify_gpu_config.py

# 2. 检查配置文件是否正确
cat VecCity-main/veccity/config/model/segment/HRNR_Hyperbolic.json | grep -A2 "gpu"

# 应该看到：
# "gpu": true,
# "gpu_id": 0,
```

### CUDA out of memory？

```bash
# 减小batch size（编辑配置文件）
# "batch_size": 32  # 从64减小
```

---

## 📈 性能提升预期

| 指标 | CPU模式（之前） | GPU模式（现在） |
|------|----------------|----------------|
| 训练时间/epoch | 2-3小时 | 5-10分钟 |
| GPU利用率 | 0% | 80-100% |
| GPU内存 | ~1.6GB | 8-12GB |
| **加速比** | 1x | **10-30x** |

---

## 📚 详细文档

- **完整指南**: [experiments/GPU_SETUP_GUIDE.md](experiments/GPU_SETUP_GUIDE.md)
- **消融实验**: [experiments/QUICK_START_ABLATION.md](experiments/QUICK_START_ABLATION.md)

---

## ✅ 验证清单

运行前确认：
- [ ] `nvidia-smi` 显示GPU可用
- [ ] `python verify_gpu_config.py` 全部通过
- [ ] 配置文件显示 `"gpu": true, "device": "cuda"`
- [ ] 准备好监控命令：`watch -n 1 nvidia-smi`

---

**立即开始：**
```bash
cd /home/user/hperroad/experiments
python verify_gpu_config.py  # 验证
./run_ablation.sh --gpu-id 0  # 运行
```

**您的GPU**: NVIDIA GeForce RTX 5070 Ti (16GB) - 完全支持！🚀
