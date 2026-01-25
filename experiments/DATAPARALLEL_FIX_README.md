# DataParallel 问题修复说明

## 问题描述

运行 `run_full_benchmark.sh` 时出现错误：
```
AttributeError: 'DataParallel' object has no attribute 'run'
```

**原因：** 当模型被 `torch.nn.DataParallel` 包装后，需要通过 `.module` 属性访问原始模型的方法。

## 解决方案

### 自动修复（推荐）

benchmark脚本已更新，会自动应用所有必要的补丁。只需重新运行：

```bash
cd ~/Mingjie/hperroad/experiments
bash run_full_benchmark.sh
```

新的脚本会在Step 0.2自动应用DataParallel修复。

### 手动修复（如果需要）

如果自动修复失败，可以手动运行修复脚本：

```bash
cd ~/Mingjie/hperroad/experiments
python fix_dataparallel.py
```

## 修复内容

脚本会修改 `VecCity-main/veccity/executor/twostep_executor.py`：

1. **添加 `_get_model()` 辅助方法**：
   ```python
   def _get_model(self):
       """Helper to get the actual model (unwrap DataParallel if needed)"""
       if isinstance(self.model, torch.nn.DataParallel):
           return self.model.module
       return self.model
   ```

2. **修复 `train()` 方法**：
   ```python
   # 修复前：
   return self.model.run(train_dataloader,eval_dataloader)

   # 修复后：
   return self._get_model().run(train_dataloader,eval_dataloader)
   ```

3. **修复 `save_model()` 和 `load_model()` 方法**：
   - 所有 `self.model.xxx` 改为 `self._get_model().xxx`
   - 确保正确访问 DataParallel 包装的模型

## 验证修复

运行benchmark后，应该看到：

```
[OPTIMIZATION] Enabling DataParallel with GPUs: [0, 1, 2, 3, 4]
[OPTIMIZATION] Model wrapped with DataParallel
```

然后训练应该正常开始，不再出现 AttributeError。

## 备份文件

修复脚本会自动创建备份：
- `twostep_executor.py.backup_dataparallel_fix`

如需恢复原始文件：
```bash
cd VecCity-main/veccity/executor
cp twostep_executor.py.backup_dataparallel_fix twostep_executor.py
```

## 预期性能

修复后，完整benchmark的预期时间：

| 阶段 | 单GPU时间 | 5-GPU时间 | 加速比 |
|------|-----------|-----------|--------|
| **预训练(100 epochs)** | ~12小时 | **3-3.5小时** | **3.4-4x** |
| **下游任务(TSI+TTE+STS)** | ~46小时 | **40-80分钟** | **35-70x** |
| **单城市总计** | ~58小时 | **4-5小时** | **11-15x** |
| **4城市总计** | ~232小时(9.7天) | **16-20小时** | **11-15x** |

## 监控运行

### 查看GPU利用率
```bash
watch -n 1 nvidia-smi
```

预期看到：
- GPU利用率：70-90%
- GPU功耗：200-240W
- 显存使用：15-20GB/GPU

### 查看运行日志
```bash
# 实时查看benchmark进度
tail -f ~/Mingjie/hperroad/experiments/results/benchmark_*/xian_tte.log

# 或查看完整日志
ls -ltr ~/Mingjie/hperroad/experiments/results/
```

## 故障排查

### 如果仍然出现AttributeError

1. 确认脚本路径正确：
   ```bash
   cd ~/Mingjie/hperroad/experiments
   python fix_dataparallel.py
   ```

2. 检查文件是否已修复：
   ```bash
   grep "_get_model" ../VecCity-main/veccity/executor/twostep_executor.py
   ```
   应该看到 `_get_model` 方法的定义。

3. 清除Python缓存：
   ```bash
   find ../VecCity-main -name "*.pyc" -delete
   find ../VecCity-main -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
   ```

### 如果GPU内存不足

编辑 `apply_downstream_optimizations.py`，减小batch_size：
```python
batch_size=512  # 改为 256 或 128
```

## 技术细节

PyTorch的 `DataParallel` 会将模型包装为：
```
DataParallel(
  (module): HRNR_Hyperbolic(...)
)
```

- `model.forward()` → 自动分发到多GPU
- `model.run()` → ❌ 不存在（自定义方法）
- `model.module.run()` → ✅ 访问原始模型的方法

`_get_model()` 辅助方法自动处理这种差异，使代码同时兼容：
- 单GPU模式：直接返回 `self.model`
- 多GPU模式：返回 `self.model.module`
