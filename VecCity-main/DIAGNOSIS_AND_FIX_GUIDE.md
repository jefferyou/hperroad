# 空间坍缩修复验证指南

## 当前状态

修复已实施但**未生效**，可能的原因：

1. **Python模块缓存** - 代码已更新但旧版本仍在内存中
2. **log_map数值问题** - 在原点附近eps可能导致切向量被不当缩小
3. **需要重新训练** - 修复只在训练时有效，无法恢复已坍缩的embedding

## 诊断步骤

### 步骤1: 检查log_map数值稳定性

运行：
```bash
python test_log_map_stability.py
```

**期望看到**:
- 如果输出显示 "发现问题：eps > 平均距离"，说明log_map在原点附近有数值问题
- 如果"范数保留比例" < 0.1，说明log_map → 平均 → exp_map 流程本身就有问题

**如果有问题**：需要修复 `hyperbolic_utils.py` 中 log_map 的eps处理

### 步骤2: 检查代码是否被加载

运行：
```bash
python test_with_fresh_init.py
```

**期望看到**:
- "✅ 代码已更新（包含log_map和exp_map）" - 说明新代码被加载
- 如果显示 "❌ 代码仍是旧版本"，需要清除Python缓存

**检查衰减比例**:
- 如果衰减比例 > 1%：修复有效
- 如果衰减比例 < 0.1%：修复无效或有其他问题

### 步骤3: 详细调试

如果步骤2显示修复无效，运行：
```bash
python debug_fix.py
```

这会显示 `_aggregate_to_cluster` 被调用时的输入/输出范数。

## 可能的修复方案

### 方案A: 修复log_map的eps问题（如果步骤1发现问题）

**问题**: 当点在原点附近时，`dist` 很小，`sinh(dist) + eps` 中的 `eps` 占主导

**修复**: 在 `hyperbolic_utils.py:129` 修改：

```python
# 当前代码
coef = dist / (sinh(dist) + eps)

# 修复为
sinh_val = sinh(dist)
# 只在sinh真的很小时才加eps
coef = dist / torch.clamp(sinh_val, min=1e-20)  # 使用更小的下限
```

### 方案B: 使用泰勒展开提升小距离时的精度

在 `hyperbolic_utils.py` 的 `log_map` 中添加：

```python
def log_map(self, x, y):
    xy = self.minkowski_dot(x, y, keepdim=True)
    eps = AdaptiveEpsilon.get_eps(x)
    xy = torch.clamp(xy, max=-1.0 - eps)

    acosh_input = -xy
    acosh_input = torch.clamp(acosh_input, min=1.0 + 1e-6)
    dist = Arcosh.apply(acosh_input)

    # 新增：当dist很小时使用泰勒展开
    # sinh(x) ≈ x + x³/6, 所以 x/sinh(x) ≈ 1 - x²/6
    small_dist_mask = dist < 1e-3

    if small_dist_mask.any():
        # 泰勒展开系数
        coef = torch.ones_like(dist)
        coef[small_dist_mask] = 1.0 - (dist[small_dist_mask]**2) / 6.0
        coef[~small_dist_mask] = dist[~small_dist_mask] / (sinh(dist[~small_dist_mask]) + eps)
    else:
        coef = dist / (sinh(dist) + eps)

    v = coef * (y + xy * x)
    return v
```

### 方案C: 使用Fréchet mean代替切空间平均

如果log_map本身有根本性问题，改用迭代的Fréchet mean（见 `fix_aggregate_to_cluster.py` 中的 `_aggregate_to_cluster_FIXED`）

### 方案D: 重新训练模型

**最彻底的方案**：

1. 确认代码已更新（运行步骤2）
2. 删除旧的模型缓存：`rm -rf veccity/cache/*/model_cache/*`
3. 重新训练模型
4. 修复后的代码将在训练过程中防止坍缩

## 快速验证清单

- [ ] 运行 `test_log_map_stability.py`
- [ ] 运行 `test_with_fresh_init.py`
- [ ] 检查"代码已更新"消息
- [ ] 检查衰减比例
- [ ] 如果需要，应用方案A或B修复log_map
- [ ] 如果仍无效，考虑方案C或D

## 预期结果

修复生效后：
- 初始嵌入空间范数: ~8.5
- 最终空间范数: >0.5 (至少保留5%)
- 时间分量: 不全是1.0
- 唯一embedding数: >50%

## 当前观察到的数值

修复前/修复无效时：
- 最终空间范数: 0.0002
- 时间分量: [1.0, 1.0]
- 衰减比例: ~2.8e-5 (0.0028%)
